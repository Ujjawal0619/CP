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

// Operator overloads <<, >>
template<typename T1, typename T2>
istream &operator>>(istream &istream, pair<T1, T2> &p) {
    return (istream >> p.first >> p.second);
} // cin >> pair<T1, T2>

template<typename T>
istream &operator>>(istream &istream, vector<T> &v) {
    for (auto &it : v) {
        cin >> it;
    }
    return istream;
} // cin >> vector<T>

template<typename T1, typename T2>
ostream &operator<<(ostream &ostream, const pair<T1, T2> &p) {
    return (ostream << p.first << " " << p.second);
} // cout << pair<T1, T2>

template<typename T>
ostream &operator<<(ostream &ostream, const vector<T> &c) {
    for (auto &it : c) {
        cout << it << " ";
    }
    return ostream;
} // cout << vector<T>

ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a,ll b) { return b?gcd(b,a%b):a;}


int main()
{
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    //code here
    ll  a, b, c, n, m, t, q, k;
    // int  a, b, c, n, m, t, q, k;
    string s;
    // int u, v;
    cin>>t;
    while(t--){
        cin>>s>>a>>b;
        
    }

    TC; return 0;
}

