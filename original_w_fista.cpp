#pragma GCC optimize("Ofast")
#pragma GCC optimize ("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
 
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <random>
#include <chrono>
#include <queue>
#include <unistd.h>
#include <cassert>
#include <iomanip>      // std::setprecision


using namespace std;


double current_density(vector<vector<pair<int, int>>>& Adj, vector<int>& weights, vector<double>& b, vector<double>& x,  long long n, long long m){
    vector<int> indices(n);
    vector<bool> deleted(n, false);
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(), [&b](const int& i, const int& j)->bool {return b[i] < b[j];});
    
    double density= 1.0*std::accumulate(x.begin(), x.end(), 0.0)/n;
    
    long long N = n;
    double M=std::accumulate(x.begin(), x.end(), 0.0);
    // cout << M << " " << accumulate(b.begin(), b.end(), 0.0) << endl;
    for(auto & i : indices){
        for(auto & [j, e_idx] : Adj[i]){
            if (!deleted[j])M-=weights[e_idx];
        }
        --N;
        // cout << M << " " << N << endl;
        if (N)density=max(density, 1.0*M/N);
        deleted[i]=true;
    }
    return density;
}



void fista(vector<vector<pair<int, int>>>& Adj, 
            vector<int>& weights, 
            long long iterations,
            long long n, 
            long long m, 
            int recalc, 
            bool provided_best_loads=false){

  vector<double> x(2*m), y(2*m), last_x(2*m), z(2*m), b(n);

  long long Delta = 0;
  for(int i=0; i<n; ++i){
    long long dd = 0;
    for(auto & [j, idx] : Adj[i]){
      dd += weights[idx];
      x[idx] = 0.5*weights[idx];
    }
    Delta = max(Delta, dd);
    b[i] = 0.5*dd;
    
  }
  double alpha = 1.0/(Delta);//0.9 for floating point issues when setting to 1/Delta for large Delta, can go slightly above 1/Delta (floating point errors) which prevents convergence
  cout << "Learning rate = " << alpha << endl;

  cout << "Before assertion=" << accumulate(b.begin(), b.end(), 0.0) << " " << accumulate(x.begin(), x.end(), 0.0) << endl;
  assert(abs(accumulate(b.begin(), b.end(), 0.0)-accumulate(x.begin(), x.end(), 0.0))<=1e-6);
  

  y = last_x = x;

  double density = current_density(Adj, weights,  b, x, n, m);

  for(int t=1; t<=iterations; ++t){
    for(int i=0; i<n; ++i){
      b[i]=0;
      for(auto & j : Adj[i]){
        b[i]+=y[j.second];
      }
      for(auto & j : Adj[i]){
        z[j.second] = y[j.second] - alpha*2.0*b[i];
      }
    }
    
    for(int i=0; i<n; ++i){
      for(auto & [j, idx] : Adj[i]){
        int sister_idx = idx % 2 ? idx-1 : idx + 1;
        x[idx] = std::clamp((z[idx]-z[sister_idx]+weights[idx])/2.0, 0.0, 1.0*weights[idx]);
        y[idx] = x[idx] + 1.0*(t-1)/(t+2) * (x[idx]-last_x[idx]);
      }
    }
    
    for(int i=0; i<2*m; ++i)last_x[i]=x[i];

    density = max(density, current_density(Adj,weights, b, x,  n, m));
    
    cout << std::setprecision(10) << "Best density=" << density << endl;
    cout << endl;
  }
}

int main(int argc, char** argv)
{    
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  cout << "Starting Graph reading" << endl;

  auto start = std::chrono::high_resolution_clock::now();
  long long n, m;
  cin >> n >> m;
  vector<vector<pair<int, int>>> Adj(n); //neighbor and edge index
  vector<int> weights(2*m, 0);
  
  int e_idx = 0;
  for(int e=0; e<m; ++e){
    int i, j, w;
    cin >> i >> j >> w;
    Adj[i].push_back({j, e_idx});
    Adj[j].push_back({i, e_idx+1});
    weights[e_idx]=weights[e_idx+1]=w;
    e_idx+=2;
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  cout << "Time for reading input is " << std::chrono::duration_cast<chrono::milliseconds	>(end - start).count() << " ms" << endl;
  
  int iters = atoi(argv[1]);
  bool print_diagnostics = true;
  int recalc = 1;
  if (argc>2)recalc=atoi(argv[2]);
  fista(Adj, weights,  iters, n, m, recalc, false);
}
