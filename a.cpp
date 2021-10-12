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
ll gcd(ll a,ll b) { return b?gcd(b,a%b):a;}


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
    cin >> n;
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    buildSegmentTree(0, 0, n-1);
    cin>>t;

    while(t--) {
        int l, r;
        cin >> l >> r;
        if(l == r) {
            cout << arr[l] << endl;
            continue;
        }
        cout << find(0, 0, n-1, l, r).second << endl;
    }
    TC; return 0;
}
