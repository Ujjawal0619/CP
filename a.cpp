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
typedef pair<int,int>              pii;
typedef vector<pair<int,int>>     vpii;
typedef double db;
const ll mod=1e9+7;
// swap, reverse(it, it)
// mt19937 mrand(random_device{}());

// int rnd(int x) { return mrand() % x;}
ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a, ll b) {
    while (b) {
        a = a % b;
        swap(a, b);
    }
    return a;
}
// static auto _ = []() { std::ios_base::sync_with_stdio(false); std::cin.tie(nullptr); return 0; }();
// #pragma GCC optimize("O3")
// #pragma GCC target("avx2, bmi, bmi2, lzcnt, popcnt")

// static const bool __boost = [](){
//     cin.tie(nullptr); 
//     cout.tie(nullptr);
//     return ios_base::sync_with_stdio(false);
// }();

vi seg(40000), arr(10000);
vpii segp(4000);

void buildSegmentTree(int i, int s, int e) {
    if(s == e) {
        segp[i].first = INT_MIN, segp[i].second = arr[e];
        return;
    }

    int mid = (s+e)/2;
    buildSegmentTree(i*2+1, s, mid);
    buildSegmentTree(i*2+2, mid+1, e);

    // on Backtracking
    vi fourVal = {segp[i*2+1].first, segp[i*2+2].first,segp[i*2+1].second, segp[i*2+2].second};
    sort(fourVal.begin(), fourVal.end(), greater<int>());
    segp[i].first = fourVal[0];
    segp[i].second = fourVal[1];
}

pair<int, int> find(int i, int s, int e, int l, int r) {
    // completely out of [l,r]
    if(e < l || s > r)
        return {INT_MIN, INT_MIN};

    // completely inside [l,r]
    if(s >= l && e <= r)
        return segp[i];

    int mid = (s+e)/2;
    pii left = find(i*2+1, s, mid, l, r);
    pii right = find(i*2+2, mid+1, e, l, r);

     vi fourVal = {left.first, right.first, left.second, right.second};
    sort(fourVal.begin(), fourVal.end(), greater<int>());
    return {fourVal[0], fourVal[1]};
}

int main() {
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // cout << "test";
    // int  a, b, c, n, m, t, q, k;
    // int u, v;
    int n, t;
    // cin >> n;
    // for(int i = 0; i < n; i++) {
    //     cin >> arr[i];
    // }
    // buildSegmentTree(0, 0, n-1);
    cin>>t;

    while(t--) {
        cout << "hello";
    }
    TC; return 0;
}

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

// Operator overloads <<, >>
template<typename T1, typename T2> // cin >> pair<T1, T2>
istream& operator>>(istream &istream, pair<T1, T2> &p) { return (istream >> p.first >> p.second); }
template<typename T> // cin >> vector<T>
istream& operator>>(istream &istream, vector<T> &v){for (auto &it : v)cin >> it;return istream;}
template<typename T1, typename T2> // cout << pair<T1, T2>
ostream& operator<<(ostream &ostream, const pair<T1, T2> &p) { return (ostream << p.first << " " << p.second); }
template<typename T> // cout << vector<T>
ostream& operator<<(ostream &ostream, const vector<T> &c) { for (auto &it : c) cout << it << " "; return ostream; }


ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a,ll b) { return b?gcd(b,a%b):a;}

const int len=50;
bool prime[len];
vi primeList;
void sieve() {
    mst(prime, 1);
    prime[0] = prime[1] = 0;
    for (int p = 2; p*p <= len; p++)  
        if (prime[p])
            for (int i = p*p; i <= len; i += p)
                prime[i] = 0;

    for(int i = 0; i < len; i++)
        if(prime[i])
            primeList.pb(i);
}

string longestCommonPrefix(vector<string> &strs) {
    
    if(strs.size() == 0) return "";

    string ans = strs[0];

    for(int i = 1; i < strs.size(); i++) {
        while(strs[i].find(ans) != 0) {
            ans = ans.substr(0, ans.size() - 1); 

            if(ans.size() == 0) {
                return "";
            }
        }
    }

    return ans;

}

void solve(int t) {
    
    vector<string> arr = {"flower", "florida", "flow"};
    cout << longestCommonPrefix(arr) << endl;
}




int main() {
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif
    ios_base::sync_with_stdio(false);  
    cin.tie(NULL);

    int t = 1;
    // cin >> t;
    while(t--) {
        solve(t+1);
        cout << '\n';
    }
    TC;

    return 0;
}






