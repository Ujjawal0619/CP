/* DP WorkShop by Vivek */
/*
	Author: Ujjawa Kumar | MANIT
	Date: 30-Jun-2024
*/


/*
	1. Question:
	You are climbing a staircase. It takes 𝑁 steps to reach the top.
	Each time you can either climb 1 or 𝑀 steps. 
	What is the minimum number of climbs you need to do to reach the top, i.e., 𝑁th stair?

	NOTE: Seems like DP, but it's greedy.
*/

int greedy() {
    return n / m + n % m;
}

/*
	2. Your task is to place eight queens on a chessboard so that no two queens are attacking each other. 
	As an additional challenge, each square is either free(.) or reserved(*), 
	and you can only place queens on the free squares. 
	However, the reserved squares do not prevent queens from attacking each other.
	NOTE: Backtraking problem, Can't be solved with DP
*/

bool check(vector<string> &b, int i, int j) {

    for(int row = i, col = j; row >= 0; row--) {
        if(b[row][col] == '#') return false;
    }

    for(int row = i, col = j; row >= 0 && col >= 0; row--, col--) {
        if(b[row][col] == '#') return false;
    }

    for(int row = i, col = j; row >= 0; row--, col++) {
        if(b[row][col] == '#') return false;
    }

    return true;
}

void rec(int i, vector<string> &b, int &ans) {
    if(i == 8) {
        ans++;
        return;
    }

    for(int j = 0; j < 8; j++) {
        if(b[i][j] != '*' && check(b, i, j)) {
            char temp = b[i][j];
            b[i][j] = '#';
            rec(i+1, b, ans);
            b[i][j] = temp;
        }
    }
}

/*
	3. Extended N-Queens
	An S-Queen is a chess piece that combines the power of a knight and a queen. 
	Find the number of ways to place N S-Queens on N x N chessboard.
*/

int n, ans;

bool check(vvi &b, int i, int j) {
    int row = i, col = j;
    // Queens Attack checking
    // curr col
    while(row >= 0) {
        if(b[row][col]) return 0;
        row--;
    }
    
    row = i, col = j;
    // left diag
    while(row >= 0 && col >= 0) {
        if(b[row][col]) return 0;
        row--, col--;
    }

    row = i, col = j;
    // right diag
    while(row >= 0 && col < n) {
        if(b[row][col]) return 0;
        row--, col++;
    }

    // Knight Attack checking
    row = i, col = j;
    // immediate upper row
    if(row-1 >= 0) {
        if(col-2 >= 0 && b[row-1][col-2]) return 0; 
        if(col+2 < n && b[row-1][col+2]) return 0; 
    }

    row = i, col = j;
    // 2nd upper row
    if(row-2 >= 0) {
        if(col-1 >= 0 && b[row-2][col-1]) return 0;
        if(col+1 < n && b[row-2][col+1]) return 0;
    }
    return 1;
}

void rec(int i, vvi &b) {
    if(i == n) {
        ans++;
        return;
    }

    for(int j = 0; j < n; j++) {
        if(check(b, i, j)) {
            b[i][j] = 1;
            rec(i+1, b);
            b[i][j] = 0;
        }
    }
}

/*
	4. You have given a positive even integer n. 
	Your task is to print all balanced parenthesis of length n in lexicographic order
*/
void rec(int n, int cnt, string &ds) {
    // pruning
    // basecase
    if(n == 0) {
        if(cnt == 0) cout << ds << endl;
        return;
    }
    // cache chick
    // check
    ds.pb('(');
    rec(n-1, cnt+1, ds);
    ds.pop_back();
    if(cnt > 0) {
        ds.pb(')');
        rec(n-1, cnt-1, ds);
        ds.pop_back();
    }
    // compute
    // save and return
}

void solve() {
    int n; cin >> n;
    string ds;
    rec(n, 0, ds);
}

/*
	5. Target sum.
*/
int n, t;
int dp[101][10001];

int rec(int level, int taken) {
	// pruning
	if(taken > t) return 0;
	//basecase
	if(level == n) {
		if(t == taken) return 1;
		else return 0;
	}

	// cache check
	if(dp[level][taken] != -1) return dp[level][taken];

	// compute
	int ans = 0;
	if(rec(level+1, taken)) ans = 1;
	else if(rec(rec+1, taken+x[level])) ans = 1;
	// save ans return
	return dp[level][taken] = ans;
}

void solve() {
	cin >> n;
	for(int i = 0; i < n; i++) cin << x[i];
	cin >> t;
	memnset(dp, -1, sizeof(dp));
	cout << rec(0, 0);
}

/*
	6. Level-up for Question 5, now you need to print the sub-sequence for each Q query having different targets t.
	NOTE: if we implement the previous same sol. we need to refresh dp array for each query as `t` is used in recurence call.
	TC: if prev solution is implemented wil be O(N*T*Q)
	Now: O(N*T + Q), since after some time most of rec call will be O(1) because of DP
*/

// Let's implement Query part.
int n, t, q;
int dp[101][10001];

int rec(int level, int left) {
	// pruning
	if(left < 0) return 0;
	//basecase
	if(level == n) {
		if(lef == 0) return 1;
		else return 0;
	}

	// cache check
	if(dp[level][left] != -1) return dp[level][left];

	// compute
	int ans = 0;
	if(rec(level+1, left)) ans = 1;
	else if(rec(rec+1, left-x[level])) ans = 1;
	// save ans return
	return dp[level][left] = ans;
}

void solve() {
	cin >> n;
	for(int i = 0; i < n; i++) cin << x[i];
	cin >> q;
	memnset(dp, -1, sizeof(dp)); // O(#S) + O(DP)
	while(q) {
		cin >> t;
		cout << rec(0, 0);
	}
	// if we go with previous sol, we need to put memset inside while loop, will incread TC by Q times.
}

/*
   Printing Part. 
   There are genrally 2 ways to print DP solution, we need to trace the transitions that lead to the final ans.
   	1. Recheck, 2. Back Pointer
   	Here will use Recheck technique.

   will increase TC since we need to traverse over all feasible solution branch this time.
   NOTICE: will not go on wrong branch this time, while make call. since solution is already cached.
*/
int n, t, q;
int dp[101][10001];

int rec(int level, int left) {
	// pruning
	if(left < 0) return 0;
	//basecase
	if(level == n) {
		if(lef == 0) return 1;
		else return 0;
	}

	// cache check
	if(dp[level][left] != -1) return dp[level][left];

	// compute
	int ans = 0;
	if(rec(level+1, left)) ans = 1;
	else if(rec(rec+1, left-x[level])) ans = 1;
	// save ans return
	return dp[level][left] = ans;
}

void printSet(int level, int left) {
	// base case
	if(level == n) return;

	// find the correct transition
	if(rec(level+1, left)) {
		cout << x[level] << " ";
		printSet(level+1, left);
	}
	else if(rec(rec+1, left-x[level])) {
		printSet(level+1, left-x[level]);
	}
}

void solve() {
	cin >> n;
	for(int i = 0; i < n; i++) cin << x[i];
	cin >> q;
	memnset(dp, -1, sizeof(dp)); // O(#S) + O(DP)
	while(q) {
		cin >> t;
		if(rec(0, t)) {
			printSet(0, t);
			cout << endl;
		} else {
			cout << "No Solution!" << endl;
		}
	}
}