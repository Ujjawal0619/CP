// Remove Duplicates:
---------------------
sort(res.begin(), res.end());
res.erase(unique(res.begin(), res.end()), res.end()); // unique returns iterator of block after last element, then from there to end block is erased.

// Eight directons:
-------------------
int row[] = {-1,-1,-1,0,0,1,1,1};
int col[] = {-1,0,1,-1,1,-1,0,1};

// Four directons:
------------------
int dirs[5] = {0, 1, 0, -1, 0};
int x = i + dirs[d], y = j + dirs[d + 1];

// Is Power Of Two:
-------------------
bool isPowerOfTow(int n){ return n && (!(n&(n-1))); }
                            //   ^-for 0  ^-- 16 & 15 = 0   =>  10000 & 01111 == 0 

// VALUE AT MSB: O(1)
--------------
int msbValue(int n) { 
    n |= n >> 1;  
    n |= n >> 2; 
    n |= n >> 4; 
    n |= n >> 8; 
    n |= n >> 16; 
    n = n + 1; 
    return (n >> 1); 
} 

// Rightmost set bit value ie 7 --> 4, 4 --> 4
    cout << (n & -n); or cout << n & ~(n - 1); // & with 2's complement of n.

// COUT SET BIT
---------------
count = 0
while (n) {
    n &= (n - 1);
    count++;
}

// BUILTIN FUN
--------------
__builtin_popcount(int) -> cont set bit                   000010001000 = 2
__builtin_ctz(int) -> return trailing zeros from right;   000001001000 = 3
__builtin_clz(int) -> return leading zeros form left:     000010010100 = 4
__builtin_parity(int) -> return true if setbits are odd   000000000111 = true

// use l && ll as postfix in function name for long && long long.
//     eg. __builtin_popcountl(x) & __builtin_popcountll(x) for long and long long

*min_element(begin(x), end(x));
*max_element(begin(x), end(x));
*minmax_element(begin(x), end(x)); // pair<int, int> min, max

next_permutation() // next permutaion of range
prev_permutation()
unique() // make unique
includes() // subsequence
rotate(v.begin(), v.begin() + 1, v.end()); // rotates by 1, use rbegin, rend for reverse rotation
for_each(nums.begin(), nums.end(), [](int &n){ n++; }); // perform certion operation on a range.

// Binary search:
lower_bound() // return iterator of value which is not less then given one.
upper_bound() // always gives iterator of greater value then given one.

/*Binary Exponintial:
power function log(n):*/
-------------------------
long long power(long long base, long long pow){
    long long res=1, b=base;
    while(pow){
        if(pow%2){// if(pow & 1)
            pow--;	// not needed
            res*=base;
        }
        else {		// not needed
            pow/=2;	// pow>>=1;
            base*=b;
        }
    }
    return res;
    /* 
    // for a^b
    ll binpow(ll a, ll b) {
        ll res = 1;
        while (b > 0) {
            if (b & 1)
                res *= a;
            a *= a;
            b >>= 1;
        }
        return res;
    }
    */
}

/*The problem with above solutions is, overflow may occur for large value of n or x. Therefore, power is generally evaluated under modulo of a large number.*/

int power(int x, unsigned int y, int p)  {  
    int res = 1;     // Initialize result  
    x = x % p; // Update x if it is more than or  
                // equal to p 
    if (x == 0) return 0; // In case x is divisible by p; 
    while (y > 0)  {  
        // If y is odd, multiply x with result  
        if (y & 1)  
            res = (res*x) % p;
        // y must be even now  
        y = y>>1; // y = y/2  also if y is odd then it reduces it by 1 while dividing. because of integer
        x = (x*x) % p;  
    }  
    return res;  
}

// C(n, r)
-------------
ll bc(ll n, ll r) { 
    ll res = 1; 
    if(n < r)
        return 0;
    if (r > n - r) r = n - r;
    
    for (ll i = 0; i < r; ++i) { 
        res *= (n - i); 
        res /= (i + 1); 
    } return res; 
}

// ADJACENCY LIST and EDGES:
----------------------------
vector<unordered_set<int> > tree( n+1 );
vector<pair<int, int> > edge;
void makeTree(int n, vector<unordered_set<int> > &tree, vector<pair<int, int> > edge){
        for(int i=0; i<n-1; i++){
        int u, v;
        cin >> u >> v;
        tree[u].insert(v);
        tree[v].insert(u);
        edge[i + 1] = make_pair(u, v);
    }
}

// BFS SSSP:
------------
vi aj[10001];
int vis[10001];
int dis[10001];

void bfs(int node){
    queue<int> q; q.push(node);
    vis[node]=1;

    while(!q.empty()){
        int curr = q.front(); q.pop();

        for(auto child: aj[curr]){
            if(vis[child]==0){
                q.push(child);
                vis[child] = 1;
                dis[child] = dis[curr]+1;
            }
        }
    }
}

// MAXIMUM DEPTH OF BINARY TREE:
-------------------------------
int maxDepth(TreeNode *root)
{
    return (root==NULL) ? 0 : max(maxDepth(root -> left), maxDepth(root -> right)) + 1;
}

// SIEVE
int len=100000;
bool prime[len]; // assuming that all are prime
vi primelist;
void sieve(){
    for (int p=2; p*p<=len; p++) // till - 100, since p*p=10000 
    { 
        if (prime[p] == false) // false means is a prime no.
        { 
            primelist.pb(p);
            for (int i=p*p; i<=len; i += p) // multiple before p*p already marked
                prime[i] = true; // declearing multiple of 'p' as not prime
        } 
    } 
}

// DISJOINT SET UNION:
----------------------
/*
    1. initialization: for all 'n' nodes the parent i.e par[i] = -1; for all 1<=i<=n.
    2. '-' -ive sign represents the parent node. other conteins its parent.
        and if parent[u] = '-x' then, x represent the rank of set (no. of nodes in set).
        here 'x' can represent the maximum node no. in set also. (x can be use for verity of pusposes)
    3. path comprassin is done during find_set call.
    4. for "union by rank": parent will be the set which has max rank,
        this will take during union operation. and parent value will also be updated.
*/

int find_set(int u, int par[]) // return the parent of set in which 'u' exist;
{   
    if(par[u] < 0)
        return u;
    return par[u] = find_set(par[u], par); // path comprassion
}

void union_set(int u, int v, int par[])
{
    u = find_set(u, par);
    v = find_set(v, par);
    if(u == v) return;

    // UNION by rank: else we can make any one of u/v as parent of other.

    if(par[u] < par[v]) // set that contains 'u' has maximum nodes, abs(par[u])
    {                   // so, valaue of par[v] is added to par[u] and u will become parent.
                        // *first updare parent value if needed the make parent.
        par[u] = par[u] + par[v]; // e.g (-5) + (-3)
        par[v] = u;
    }
    else
    {
        par[v] = par[v] + par[u];
        par[u] = v;
    }
}

// SEGMENT TREE:
----------------
    1. leaf represents the arr values.
    2. size of segment tree is <= 4*(no. of elemnts in arr).
    3. range query takes log(n), point update take log(n).
    4. in 0 based index Lchild = 2*si + 1, Rchild = 2*si + 2.
    5. first the leaf node value is calculated and using leaf node 
       we creat upper parent node using its both cilds value.
/*
    si = segment tree index (current).
    ss = segment starting index.
    se = segment ending index.
    stree = segment tree.
    arr = elemnts form which segment tree is made.
*/
void build_stree(int si, int ss, int se, int stree[], int arr[])
{
    if(ss == se) // single element
        {stree[si] = arr[ss]; return;} // filling leaves from arr value.

    int mid = (ss + se)/2;

    build_stree(2*si + 1, ss, mid, stree, arr); // left tree call
    build_stree(2*si + 2, mid+1, se, stree, arr); // right tree call

    // as left & right tree is calculated so we can calculate current node/element
    // using its left & right child.
    stree[si] = min (stree[2*si +1], stree[2*si +2]); 
    //   parent         L child         R child
}   // call: build_stree(0, 0, n-1, stree, arr);


int query(int si, int ss, int se, int l, int r, int stree[]){
    // (l<=r) always
    // if segment tree [ss, se] either left or right only of range[l,r].
    if(l > se || r < ss)
        return INT_MAX;

    // if segment tree range[ss, se]  completely inside range[l,r]
    if(l <= ss && r >= se)
        return stree[si];

    // partial overlapping
    int mid = (ss + se)/2;
    int ansltree = query(2*si +1, ss, mid, l, r, stree);
    int ansrtree = query(2*si +2, mid+1, se, l, r, stree);

    return min(ansltree, ansrtree);
}// call: cout<<query(0, 0, n-1, u, v, stree)<<'\n';

// if value of arr is updated then we need to update segment tree also.
void update(int si, int ss, int se, int qi, int stree[], int arr[])
{
    // qi -> index of arr whare value is updated.
    if(ss == se)// reaching to leaf, which represent arr[qi] value.
    {
        stree[si] = arr[ss]; return;
    }

    int mid = (ss + se)/2;
    if(qi <= mid) update(2*si + 1, ss, mid, qi, stree, arr);
    else          update(2*si + 2, mid + 1, se, qi, stree, arr);
    // updating all parent bottom to up while returning.
    stree[si] = min(stree[2*si + 1], stree[2*si + 2]);
}

// for lazy propagation include the following code inside update query etc:

if(lazy[si] != 0)
{
    int dx = lazy[si];
    lazy[si] = 0;
    st[si] = dx*(se - si + 1); // for all leaf nodes of current tree

    if(ss!= se)
        lazy[2*si+1] += dx, lazy[2*si+2] +=dx;
}