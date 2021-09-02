// Author: Ujjawal Kumar, MANIT
#include <bits/stdc++.h>
using namespace std;
#define rep(i,a,n) for (int i=a;i<n;i++)
#define per(i,n,a) for (int i=n;i>=a;i--)
#define pb push_back
#define mp make_pair
#define all(x) (x).begin(),(x).end()
#define fst first
#define snd second
#define SZ(x) ((int)(x).size())
#define endl '\n'
#define TC cerr<<"Time elapsed: "<<1000*clock() /CLOCKS_PER_SEC <<"ms: ";
#define mst(a, b) memset(a, b, sizeof(a)); // b can be 0 or -1 only
#define minv(x) *min_element(all(x));
#define maxv(x) *max_element(all(x));
typedef long long ll;
typedef vector<int>               vi;
typedef vector<ll>                vll;
typedef vector<vi>                vvi;
typedef vector<vll>               vvll;
typedef vector<pair<int,int>>     vpii;
typedef double db;
const ll mod=1e9+7;
// swap, reverse(it, it)
// mt19937 mrand(random_device{}());

// int rnd(int x) { return mrand() % x;}
ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a,ll b) { return b?gcd(b,a%b):a;}

void heapify(vector<int> &arr, int i) {
    int n = arr.size();
    int lChild = i*2 + 1;
    int rChild = i*2 + 2;
    int largest = i; // assume parent is largest

    if(lChild < n && arr[largest] < arr[lChild]) largest = lChild; // left child is larger
    if(rChild < n && arr[largest] < arr[rChild]) largest = rChild; // right child is larger

    if(i != largest) { // there is a change at ith index (left or right child), so we need 
        swap(arr[largest], arr[i]); // fix the parent
        heapify(arr, largest); // fix further either of left or right child
    }
}

void makeHeap(vector<int> &arr) {
    int n = arr.size();

    for(int i = n/2 - 1; i >= 0; i--) {
        heapify(arr, i);
    }
}


void isSubsetSum(vector<int> &set, int n, int sum, long long &count) {
    // cout << n << " " << sum << endl;
    bool subset[n + 1][sum + 1];

    for (int i = 0; i <= n; i++)
        subset[i][0] = true;
 
    for (int i = 1; i <= sum; i++)
        subset[0][i] = false;
 
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= sum; j++) {
            if (j < set[i - 1])
                subset[i][j] = subset[i - 1][j];
            if (j >= set[i - 1])
                subset[i][j] = subset[i - 1][j]
                               || subset[i - 1][j - set[i - 1]];
        }
    }
 
       // uncomment this code to print table
     // for (int i = 0; i <= n; i++)
     // {
     //   for (int j = 0; j <= sum; j++) {
     //        printf ("%4d", subset[i][j]);
     //   }
     //   printf("\n");
     // }

    for(int i = n; i >=0; i--) {
        // cout<<subset[sum][i]<<endl;
        if(subset[i][sum]) {
            count = (++count) % mod;
        }
    }

}


int main()
{
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    //code here
    // int  a, b, c, n, m, t, q, k;
    // int u, v;
    // cin>>t;
    // while(t--){
    //     vi arr;
    //     cin >> n;
    //     rep(i,0,n) {
    //         cin >> a;
    //         arr.pb(a);
    //     }

    //     makeHeap(arr);
    //     vector<int>res;
    //     while(arr.size()>0)
    //     {
    //         res.push_back(arr[0]);
    //         arr.erase(arr.begin()+0);
    //         makeHeap(arr);

    //     }
    //     rep(i,0,res.size())
    //     {
    //         cout<<res[i]<<" ";
    //     }

    // }
   
    int t;
    long long count = 0;
    cin>>t;
    vector<int> set;
    while(t--){
        int a, b;
        cin >> a >> b;
        
        if(a == 0) {
            set.push_back(b);
        } else if (a == 1) {
            set.erase(find(set.begin(), set.end(), b));
        } else {
            isSubsetSum(set, set.size(), b, count);
            cout<< count << " ";
            count = 0;
        }
    }

    TC; return 0;
}

