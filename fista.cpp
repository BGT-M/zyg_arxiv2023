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

#if PARALLEL 
#include <omp.h>
#endif 

using namespace std;
#include "utilities.h"


// vector<double> get_initial(vector<vector<pair<int, int>>>& Adj, long long n,  long long m){
//     vector<double> x(2*m, 0.0);

//     priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>> > pq;
//     for(int i=0; i<n; ++i)pq.push({Adj[i].size(), i});
//     vector<bool> deleted(n, false);
//     vector<int> degree(n, 0);
//     for(int i=0; i<n; ++i)degree[i]=Adj[i].size();
//     long long N = n;
//     long long M = m;
//     double density = 1.0*M/N;
//     while(!pq.empty()){
//         auto [d, i] = pq.top();
//         pq.pop();
//         if (deleted[i])continue;
        
//         for(auto & [j, e_idx] : Adj[i]){
//             if ( !deleted[j] ){
//                 if (i!=j){
//                     pq.push({--degree[j], j});
//                     M--;
//                 }
//                 x[e_idx]=1;
//             }
//         }
//         N--;
//         if (N>0){
//             density = max(density, 1.0*M/N);
//         }
//         deleted[i] = true;
//     }
//     // assert (abs(accumulate(x.begin(), x.end(), 0)-m)<=1e-5 );
//     return x;
// }

// double current_density(vector<vector<pair<int, int>>>& Adj, vector<double>& b, vector<double>& x,  long long n, long long m, long long t=1){
//     priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>> > pq;

//     for(int i=0; i<n; ++i)pq.push({b[i]+Adj[i].size(), i});
//     vector<bool> deleted(n, false);
//     vector<double> degree(n, 0);
//     for(int i=0; i<n; ++i)degree[i]=b[i]+Adj[i].size();
//     long long N = n;
//     long long M = m;
//     double density = (double)(1.0*M/N);
//     while(!pq.empty() and N>1){
//         auto [d, i] = pq.top();
//         pq.pop();
//         if (deleted[i])continue;
//         // cout << i << endl;
//         for(auto & [j, e_idx] : Adj[i]){
//             int sister_idx = e_idx % 2 ? e_idx-1 : e_idx + 1;
//             if ( !deleted[j] ){
//               pq.push({degree[j] - t * x[sister_idx]-1, j});
//               degree[j] -= (t*x[sister_idx]+1);
//               M--;
//             }
//         }
//         N--;
//         density = max(density, 1.0*M/N);
//         deleted[i] = true;
//     }
//     // assert(abs(M)<1e-5);
//     return density;
// }

vector<double> b_star;

void fista(vector<vector<pair<int, int>>>& Adj, 
            long long iterations,
            long long n, 
            long long m, 
            int recalc, 
            bool provided_best_loads=false){


  long long Delta = max_element(Adj.begin(), Adj.end(), [](vector<pair<int, int>>& v1, vector<pair<int, int>>& v2)->bool {return v1.size()<v2.size();})->size();
  double alpha = 0.9/(Delta);//0.9 for floating point issues when setting to 1/Delta for large Delta, can go slightly above 1/Delta (floating point errors) which prevents convergence
  cout << "Learning rate = " << alpha << endl;
  vector<double> x(2*m), y(2*m), last_x(2*m), z(2*m), b(n);

  y = last_x = x = get_initial(Adj, n, m);
  #if PARALLEL 
  #pragma omp parallel for
  #endif 
  for(int i=0; i<n; ++i){
    for(auto & j : Adj[i]){
      b[i]+=y[j.second];
    }
  }
  double density = current_density(Adj, b, x, n, m);
  double density_sorting = 0.0;
  double tk = 1, tknew = 1;
  double fxk = 0;
  for(int i=0; i<n; ++i)fxk += b[i]*b[i];

  for(int t=1; t<=iterations; ++t){
    auto start1 = std::chrono::high_resolution_clock::now();

    #if PARALLEL 
    #pragma omp parallel for
    #endif 
    for(int i=0; i<n; ++i){
      b[i]=0;
      for(auto & j : Adj[i]){
        b[i]+=y[j.second];
      }
      for(auto & j : Adj[i]){
        z[j.second] = y[j.second] - alpha*2.0*b[i];
      }
    }

    tknew = (1.0+sqrt(1+4*tk*tk))/2.0;
    
    #if PARALLEL 
    #pragma omp parallel for
    #endif 
    for(int i=0; i<n; ++i){
      for(auto & [j, idx] : Adj[i]){
        int sister_idx = idx % 2 ? idx-1 : idx + 1;
        x[idx] = clamp((z[idx]-z[sister_idx]+1)/2.0, 0.0, 1.0);
        y[idx] = x[idx] + 1.0*(tk-1)/(tknew) * (x[idx]-last_x[idx])  + tk/tknew * (x[idx]-y[idx]);
      }
    }
    tk = tknew; 
    
    #if PARALLEL 
    #pragma omp parallel for
    #endif 
    for(int i=0; i<2*m; ++i)last_x[i]=x[i];

    
    auto finish1 = std::chrono::high_resolution_clock::now();

    /*
    Print diagnostics for experimental section, can safely remove this for non experiments. 
    */
    cout << "Iteration=" << t << endl;
    cout << "Time=" << std::chrono::duration_cast<chrono::milliseconds	>(finish1 - start1).count() << " miliseconds" << endl;
    if (provided_best_loads){
      double E = 0;
      for(int i=0; i<n; ++i){
        E += (b[i] - b_star[i])*(b[i] - b_star[i]);
      }
      E = sqrt(E);
      cout << "Error=" << E << endl;
    }
    for(int i=0; i<n; ++i)b[i]*=t;
    if (t%recalc==0)density = max(density, current_density(Adj, b, x,  n, m, t));
    
    cout << "Best density=" << density << endl;
    density_sorting = max(density_sorting, current_density_sorting(Adj, b, n, m));
    cout << "Sorting density=" << density_sorting << endl;

    double sum_of_squares = 0.0;
    for(int i=0; i<n; ++i)sum_of_squares+=b[i]*b[i]/(t*t);
    sum_of_squares = sqrt(sum_of_squares);
    cout << "Sum of squares=" << sum_of_squares << endl;
    if (t+1>=iterations){
      cout << "b vector=[";
      for(int i=0; i<n; ++i)cout << b[i]/t << ",";
      cout << "]" << endl;
    }

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

  
  int e_idx = 0;
  for(int e=0; e<m; ++e){
    int i, j;
    cin >> i >> j;
    Adj[i].push_back({j, e_idx});
    Adj[j].push_back({i, e_idx+1});
    e_idx+=2;
  }
  string line;
  getline(cin, line); // read \n 

  bool provided_best_loads = false;
  b_star = vector<double>(n, 0); //optimal load vector 
  int i = 0;
  while(getline(cin, line)){
    provided_best_loads = true;
    b_star[i++] = stod(line); 
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  cout << "Time for reading input is " << std::chrono::duration_cast<chrono::milliseconds	>(end - start).count() << " ms" << endl;
  
  int iters = atoi(argv[1]);
  bool print_diagnostics = true;
  int recalc = 1;
  if (argc>2)recalc=atoi(argv[2]);
  fista(Adj,  iters, n, m, recalc, provided_best_loads);
}
