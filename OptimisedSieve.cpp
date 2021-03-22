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
#define sz(x) ((int)(x).size())
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


const int N = 1e8;

int main()
{
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    //code here
 
    // cin>>t;
    // while(t--){
        
    // }

    // int lowestPrime[N+1] = {0};
    vector<int> lowestPrime(N+1, 0);
    vector<int> prime;

    for(int i=2; i<N; i++) {
        if(lowestPrime[i] == 0) {
            lowestPrime[i] = i;
            prime.pb(i);
        }

        for(int j=0; j < prime.size()
                     && prime[j] <= lowestPrime[i]
                     && i*prime[j] <= N;
                     j++) {
            lowestPrime[i*prime[j]] = prime[j];
        }
    }

    for(int i=0; i<prime.size(); i += 100)
        cout<<prime[i]<<endl;
        // printf("%d\n", prime[i]);
    
    TC; return 0;
}
