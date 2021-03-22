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

int vis[4002];
int bx;
int tbl[4002][4002];
bool flag;

void display(vi &v, int k) {
    int sum = 0, t1 = 0, t2 = 0;
    for(int i=v.size()-1; i>=0; i--) {
        if(vis[i] == 0 && sum < k) {
            sum += v[i];
            t2++;
        }
        else if(vis[i] == 1){
            cout<<v[i]<<" ";
            t1++;
        }
    }
    if(sum >= k)
        flag = true;
    bx = min(bx, t2 + t1);
    // cout<<"box: "<<t1<<':'<<t2;
    cout<<endl;
}

bool solve(vi &box, int n, int k, int sum) {
    if(n <= 0)
        return 0;
    if(sum >= k) {
        display(box, k);
        return 1;
    }
    bool a, b;
    vis[n-1] = 1;
    if(tbl[n][min(k, sum + box[n-1])] != -1)
        a = tbl[n][min(k, sum + box[n-1])];
    else
        a = tbl[n][min(k, sum + box[n-1])] = solve(box, n-1, k, sum + box[n-1]);
    vis[n-1] = 0;
    if(tbl[n][min(k, sum)] != -1)
        b = tbl[n][min(k, sum)];
    else
        b = tbl[n][min(k, sum)] = solve(box, n-1, k, sum);
    return a | b;
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
    int  a, b, c, n, m, t, q, k;
    // int u, v;
    cin>>t;
    while(t--){
        cin>>n>>k;
        vi boxHeight(n);
        bx = INT_MAX;
        flag = false;
        mst(tbl, -1);
        a = 0;
        rep(i,0,n) {
            cin>>boxHeight[i];
            a += boxHeight[i];
        }
        if(a < 2*k) return -1;
        sort(boxHeight.begin(), boxHeight.end());
        solve(boxHeight, n, k, 0);
        if(flag)
            cout<<bx<<endl;
        else
            cout<<-1<<endl;
    }
    TC; return 0;
}

