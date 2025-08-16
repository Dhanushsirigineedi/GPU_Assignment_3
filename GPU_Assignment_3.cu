#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <climits>

#define MOD 1000000007

using std::cin;
using std::cout;
using std::vector;
using std::string;

// Structure representing an edge in the graph with source, destination, weight, and a label.
struct Edge {
    int src, dest, weight,label;
};

// Kernel to modify edge weights based on the label value.
__global__ void modify_weights(Edge *edges,int E)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=E)return;
    edges[idx].weight*=edges[idx].label;
}

// Kernel to initialize each vertex's parent to itself for the union-find data structure.
__global__ void initialize_Parent(int *parent, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V) return;
    parent[idx] = idx;
}

// Kernel to initialize the array holding the minimum edge for each component.
__global__ void initialize_min_edges(unsigned long long int *d_min_edges, int V,int* d_flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V) return;
    // Combine INT_MAX as the weight (upper 32 bits) and 0xFFFFFFFF as an invalid index (lower 32 bits)
    d_min_edges[idx] = ((unsigned long long int)INT_MAX << 32) | 0xFFFFFFFF;
    // Reset the flag on the first thread to indicate work is pending
    if(idx==0)atomicExch(d_flag, 0);
}

// Device function to find root of a node with optional path compression
__device__ int find_the_root(int u, int *parent, bool compressflag) {
    //used customized situation to compress to avoid the datarace condition
    if(compressflag)
    {
        int root = u;
        while (parent[root] != root) root = parent[root];
        while (parent[u] != root)
        {
            int temp = parent[u];
            parent[u] = root;
            u = temp;
        }
        return root;
    }
    else
    {
        while (parent[u] != u)
        {
            parent[u] = parent[parent[u]];
            u = parent[u];
        }
        return u;
    }
}

// Kernel to find the minimum edge for each component(each tread works with an edge)
__global__ void finding_min_edges(Edge *edges, int E, int *parent, unsigned long long int *d_min_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    //extracting the details of the edge
    Edge e = edges[idx];
    int u = e.src;
    int v = e.dest;

    //finding the root for both the ends of an edge
    int root_of_u = find_the_root(u, parent,1);
    int root_of_v = find_the_root(v, parent,1);
    if(root_of_u == root_of_v)return;//same component(previously merged)
    //wrapping the edge weight & edge id together and modifying min edge for a component
    unsigned long long int temp_val = ((unsigned long long int)e.weight << 32) | idx;
    atomicMin(&d_min_edges[root_of_u], temp_val);
    atomicMin(&d_min_edges[root_of_v], temp_val);
}

// Kernel to add the selected edges to the MST and update parent array i.e.,merging the components
__global__ void merge_the_components(Edge *edges, unsigned long long int *d_min_edges, int *parent, int *mst_weight,int* flag, int V)
{
    int root = blockIdx.x * blockDim.x + threadIdx.x;
    if (root >= V)return;

    //extracting edge id,weight from the wrapped values in the array
    unsigned long long int min_edge_val = d_min_edges[root];
    int edge_idx = (int)(min_edge_val & 0xFFFFFFFF);
    int weight = (int)(min_edge_val >> 32);

    if (weight == INT_MAX) return;//not modified

    Edge e = edges[edge_idx];
    int u = e.src;
    int v = e.dest;
    int wt=e.weight;

    //finding the roots
    int rootU = find_the_root(u, parent,0);
    int rootV = find_the_root(v, parent,0);
    //running the while loop till both are merged
    //used the atomic instructions to avoid datarace
    while(rootU!=rootV)
    {
        if (rootU < rootV)
        {
            if (atomicCAS(&parent[rootU], rootU, rootV) == rootU)
            {
                atomicAdd(mst_weight, wt % MOD);
                atomicExch(flag, 1);
                break;
            }
        }
        else
        {
            if (atomicCAS(&parent[rootV], rootV, rootU) == rootV)
            {
                atomicAdd(mst_weight, wt % MOD);
                atomicExch(flag, 1);
                break;
            }
        }
        rootU = find_the_root(u, parent,0);
        rootV = find_the_root(v, parent,0);
    }
}

//main function
int main()
{
    int V;
    cin >> V;
    int E;
    cin >> E;
    vector<Edge> edges;

    dim3 blockDim(1024);//dimension for a block
    dim3 gridDim_vertex((V + blockDim.x - 1) / blockDim.x);//grid dimension when V(vertex) number of threads are launched
    dim3 gridDim_edge((E + blockDim.x - 1) / blockDim.x);//grid dimension when E(Edge) number of threads are launched

    //defining the pointers and allocating the memory for device arrays and variables
    Edge *d_edges;
    cudaMalloc(&d_edges, E * sizeof(Edge));
    int *d_parent;
    cudaMalloc(&d_parent, V * sizeof(int));
    unsigned long long int *d_min_edges;
    cudaMalloc(&d_min_edges, V * sizeof(unsigned long long int));
    int *d_mst_weight, *d_flag;
    cudaMalloc(&d_mst_weight, sizeof(int));
    cudaMemset(d_mst_weight, 0, sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 1, sizeof(int));

    //taking the input and giving labels to them based on the type of road
    for (int i = 0; i < E; i++)
    {
        int u, v, wt;
        string s;
        cin >> u >> v >> wt;
        cin >> s;
        int label=1;
        if(s=="green")label=2;
        else if(s=="dept")label=3;
        else if(s=="traffic")label=5;
        edges.push_back({u, v, wt,label});
    }

    //copying the data from host array to device array
    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);

    /*****************start of the timing block*****************/
    auto start = std::chrono::high_resolution_clock::now();

    //kernel launch for modifying the weights
    modify_weights<<<gridDim_edge,blockDim>>>(d_edges,E);

    //kernel launch to intialize the parent array
    initialize_Parent<<<gridDim_vertex, blockDim>>>(d_parent, V);
    //cudaDeviceSynchronize();
    
    int h_flag,max_iter = log2(V) + 1;//max iterations it can take to merge all in extreme case

    for (int iter = 0; iter < max_iter; ++iter)
    {
        cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if(h_flag==0)break;//stop if all components are merged
        initialize_min_edges<<<gridDim_vertex, blockDim>>>(d_min_edges, V, d_flag);
        //cudaDeviceSynchronize();
        finding_min_edges<<<gridDim_edge, blockDim>>>(d_edges, E, d_parent, d_min_edges);
        //cudaDeviceSynchronize();
        merge_the_components<<<gridDim_vertex, blockDim>>>(d_edges, d_min_edges, d_parent, d_mst_weight,d_flag, V);
        //cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    /*****************end of the timing block*****************/

    //copying the answer to a host side variable and do required computations
    int h_mst_weight;
    cudaMemcpy(&h_mst_weight, d_mst_weight, sizeof(int), cudaMemcpyDeviceToHost);
    h_mst_weight %= MOD;
    if (h_mst_weight < 0) h_mst_weight += MOD;

    cout << h_mst_weight << "\n";

    //free the cuda memory allocated
    cudaFree(d_edges);
    cudaFree(d_parent);
    cudaFree(d_min_edges);
    cudaFree(d_mst_weight);

    // std::ofstream file("cuda.out");
    // if (file.is_open())
    // {
    //     file << h_mst_weight;
    //     file.close();
    // }
    // else
    // {
    //     std::cout << "Unable to open file";
    // }

    // std::ofstream file2("cuda_timing.out");
    // if (file2.is_open())
    // {
    //     file2 << elapsed1.count() << "\n";
    //     file2.close();
    // }
    // else
    // {
    //     std::cout << "Unable to open file";
    // }

    return 0;
}