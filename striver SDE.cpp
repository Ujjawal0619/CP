DAY 1: Arrays
=============

1. // Sort an array of 0’s 1’s 2’s without using extra space or sorting algo (LC)
----------------------------------------------------------------------------
method 1: sort
mehtod 2: count
method 3: Dutch National Flag Algorithm
void sortColors2(vector<int>& a) {
    int low = 0, mid = 0, hi = a.size() - 1;
    
    while(mid <= hi) {
        switch(a[mid]) { // three important case of dutch national flag algo
            case 0:
                swap(a[low++], a[mid++]);
                break;
            case 1:
                mid++;
                break;
            default:
                swap(a[mid], a[hi--]);
        }
    }
}

2. // Repeat and Missing Number (IB)
-------------------------------
vector<int> findErrorNums(vector<int>& nums) {
    // METHOD 1: 
    /* let 'a', 'b' are repeated, missing number respectively */
    long long n = nums.size();
    
    long long sum = n*(n + 1)/2;
    long long sqrsum = n*(n + 1)*(2*n + 1)/6;
    
    for(int ele: nums) {
        sum -= ele;
        sqrsum -= pow(ele, 2);
    }
    // sum contains (a - b)                             --- (1)
    // sqrsum contains (a^2 - b^2) == (a + b)(a - b)    --- (2)

    int t = sqrsum / sum; // (a + b)                    --- (3)
    int a = (t + sum)/2;  // from  (1) + (2)
    int b = (t - sum)/2;  // from  (1) - (2)
    
    return {a, b};

    //METHOD 2:
    int xor2 = 0, xor0 = 0, xor1 = 0;
    for (int n: nums) xor2 ^= n;
    for (int i = 1; i <= nums.size(); i++) xor2 ^= i; // contains xor of tow required elements
    
    // rightmost set bit can be 0/1 in 1st and 1/0 in 2nd
    int rightmostbit = xor2 & ~(xor2 - 1); // or xor2 & -xor2, will be always power of two
    
    // filtering possibilities
    for (int n: nums)
        if ((n & rightmostbit)) // if set in n
            xor1 ^= n;
        else
            xor0 ^= n;

    for (int i = 1; i <= nums.size(); i++)
        if ((i & rightmostbit))
            xor1 ^= i;
        else
            xor0 ^= i;
    
    // ordering the ans as (repeating, missing)
    for (int i = 0; i < nums.size(); i++)
        if (nums[i] == xor0)
            return {xor0, xor1};
    
    return {xor1, xor0};

    // METHOD 3: using frequency array, missing will be 0 and repting will be 2 count.
}

3. // Merge two sorted Arrays without extra space 
-------------------------------------------------
method 1: store and sort, then place in order. TC: O(nlog(n)) SC: O(n)
method 2: insertion sort technique TC: O(nm) SC: O(1)
void merge(int X[], int Y[], int m, int n) {
    /* Consider each element `X[i]` of array `X` and ignore the element if it is
     already in the correct order; otherwise, swap it with the next smaller
     element, which happens to be the first element of `Y` always.*/
    for (int i = 0; i < m; i++) {
        // compare the current element of `X[]` with the first element of `Y[]`
        if (X[i] > Y[0]) {
            swap(X[i], Y[0]);
            int first = Y[0];
 
            // move `Y[0]` to its correct position to maintain the sorted
            // order of `Y[]`. Note: `Y[1…n-1]` is already sorted
            int k = 1;
            while (k < n && Y[k] < first) {
                Y[k - 1] = Y[k];
                k++;
            }
 
            Y[k - 1] = first;
        }
    }
}

method 3: Gap Method TC: O(nlog(n)) SC: O(1)
int nextGap(int gap) {
    if (gap <= 1) return 0;
    return (gap / 2) + (gap % 2);
}
void merge(int* arr1, int* arr2, int n, int m) {
    int i, j, gap = n + m;
    for (gap = nextGap(gap);
         gap > 0; gap = nextGap(gap))
    {
        // comparing elements in the first array.
        for (i = 0; i + gap < n; i++)
            if (arr1[i] > arr1[i + gap])
                swap(arr1[i], arr1[i + gap]);
 
        // comparing elements in both arrays.
        for (j = gap > n ? gap - n : 0;
             i < n && j < m;
             i++, j++)
            if (arr1[i] > arr2[j])
                swap(arr1[i], arr2[j]);
 
        if (j < m) {
            // comparing elements in the second array.
            for (j = 0; j + gap < m; j++)
                if (arr2[j] > arr2[j + gap])
                    swap(arr2[j], arr2[j + gap]);
        }
    }
}

4. // Find max sum sub array
----------------------------
method 1: three nested loop, two for sub array and third for sum(i to j);
method 2: eleminate 3rd sum loop to reduce complexity to n^2 use 2nd for addition
method 3: Kadane’s Algorithm

void kadane(vector<int> arr) {
    int sum = 0, mx = INT_MIN;

    for(auto ele: arr) {
        sum += ele;
        mx = max(mx, sum);
        sum = sum < 0 ? 0:sum;
    }
    return mx;
} 

5. // Merge intervals 
method 1: sort and traverse for all overlaped interval for each interval and merge them; TC O(n^2) SC O(n)
method 2: TC O(n) SC O(n)

vector<vector<int>> merge(vector<vector<int>> intervals) {
    vector<vector<int>> ans; vector<int> temp; // [a,b]
    if(intervals.size() == 0) return {};

    sort(intervals.begin(), intervals.end());
    temp = intervals[0];

    for(auto it: intervals) { // it -> [p,q]
        if(it[0] <= temp[1]) // if first number of current range lies in between temp
            temp[1] = max(it[1], temp[1]);
        else { // starting value of current range is > end of temp
            ans.push_back(temp);
            temp = it;
        }
    }
    ans.push_back(temp);
    return ans;
}

6. // Find the duplicate in an array of N+1 integers.
method 1: sort and check i, i+1 are same 
mehtod 2: use hashing array
mehtod 3: Tortoise method
int findDuplicate(vector<int> nums) {
    int slow = nums[0], fast = nums[0];
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while(slow != fast);

    fast = nums[0];
    while(slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    return slow;
}

DAY 2: Arrays
=============
1. // Set matrix to 0
method 1: brute
method 2: use helping array one for row and one for column
method 3: instead of seperate helping arry use first row & col of matrix *taking care of first cell.
void setZeroes(vector<vector<int>> &mat) {
    int col0 = 1, m = mat.size(), n = mat[0].size();

    for(int i = 0; i < m; i++) {
        if(mat[i][0] == 0) col0 = 0;
        for(int j = 1; j < n; j++)
            if(mat[i][j] == 0)
                mat[i][0] = mat[0][j] = 0;
    }

    for(int i = m-1; i >= 0; i--) {
        for(int j = n-1; j >=1; j--)
            if(mat[i][0] == 0 || mat[0][j] == 0)
                mat[i][j] = 0;
        if(col0 == 0) mat[i][0] = 0;
    }
}

2. // pascal traingle
-------------------
/*
    there can be of three type. 
    1. make whole traingle O(n^2)
    2. find any one whose row/col is given -> calculate (r-1)C(c-1); O(n)
    3. find nth row of pascle: use patter of multiplying/dividing. O(n)
*/
    // 1. sol:
    vvi generate(int numRows) {
        vvi r(numRows);

        for(int i = 0; i < numRows; i++) {
            r[i].resize(i+1);
            r[i][0] = r[i][i] = 1;

            for(int j = 1; j < i; j++)
                r[i][j] = r[i - 1][j - 1] + r[i - 1][j];
        }
        return r;
    }
    // 2. sol:
    int binomialCoeff(int n, int k) {
        int res = 1;
     
        // Since C(n, k) = C(n, n-k)
        if (k > n - k)
            k = n - k;
     
        // Calculate value of
        // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
        for (int i = 0; i < k; ++i) {
            res *= (n - i);
            res /= (i + 1);
        }
        return res;
    }

3. // next permutation
----------------------
method 1: use inbuild function
method 2: generate all permutation find given and the next will be ans.
method 3:

vector<int> nextPermutaion(vector<int> &arr) {
    int n = arr.size(), k, l;

    for(k = n - 2; k >= 0; k--)
        if(arr[k] < arr[k+1]) break;
    
    if(k < 0) reverse(arr.begin(), arr.end());
    else {
        for(l = n - 1; l > k; l--)
            if(arr[l] > arr[k]) break;
        swap(arr[k], arr[l]);
        reverse(arr.begin() + k + 1, arr.end()); // reversing forom k+1 (peak) till end;
    }
}

4. // Count Inversion in an array. i.e (a[i] > a[j] && i < j)
---------------------------------
method 1: use two for loops and cont accordingly
mehtod 2: use merge sort algorithm

int merge(vector<int> &arr, vector<int> &temp, int left, int mid, int right) {
    int i, j, k;
    int inv_count = 0;

    i = left, j = mid, k = left;
    while((i <= mid-1) && (j <= right)) {
        if(arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            // this is tricky
            inv_count += (mid - i);
        }
    }

    while(i <= mid - 1)
        temp[k++] = arr[i++];

    while(j <= right)
        temp[k++] = arr[j++];

    for(i = left; j <= right; i++)
        arr[i] = temp[i];

    return inv_count;
}

int mergeSort(vector<int> arr, vector<int> &temp, int left, int right) {
    int mid, inv_count = 0;
    if(right > left)  {
        mid = (left + right) / 2;

        inv_count += mergeSort(arr, temp, left, mid);
        inv_count += mergeSort(arr, temp, mid + 1, right);

        inv_count += merge(arr, temp, left, mid + 1, right);
    }
    return inv_count;
}

5. // Best time to buy and sell stocks
--------------------------------------
method 1: use two nested for loop and find max profit
method 2: keep track of min price and each next element find max profit.
int buyAndSell(vector<int> &price) {
    int maxProfit = 0, currMin = INT_MAX;

    for(auto ele: price) {
        currMin = min(currMin, ele);
        maxProfit = max(maxProfit, ele - currMin);
    }
    return maxProfit;
}

6. // Rotate Matrix/image with 90 degree.
-----------------------------------------
method 1: traverse and copy in other matrix
method 2: use two pointer and rotate around. (see leetcode);
void rotate(vector<vector<int>>& matrix) {
    int a = 0, b = matrix.size()-1;
    
    while(a<b) {
        for( int i = 0; i < b-a; i++ ) {
            swap(matrix[a][a+i], matrix[a+i][b]);
            swap(matrix[a][a+i], matrix[b][b-i]);
            swap(matrix[a][a+i], matrix[b-i][a]);
        }
        a++, b--;
    }
}
method 3: transpose and reverse all rows. (super simple)


DAY 3: Array/Maths
==================
1. // Search in sorted matrix.
------------------------------
// two varient 
method 1: perform a linear search.
method 2: 
// 1) gfg (last elemet of ith row will (may or may not) first element of i+1 th row)
start from right top most corner cell. and search greddly. down or left only.
bool search(int mat[][], int n, int m, int x) {
    int i = 0, j = m - 1;
    while(i < n && j >= 0) {
        if(mat[i][j] == x)
            return true;
        if(mat[i][j] > x) j--;
        else i++;
    }
}

// 2) leetcode (last elemet of ith row will <= first element of i+1 th row always)
use Binary Search. convert range into index values using /,% operator.
 bool searchMatrix(vvi &mat, int t) {
    it(!mat.size()) return 0;
    int n = mat.size(), m = mat[0].size();
    int lo = 0, hi = n*m - 1;

    while(lo <= hi) {
        int mid = lo + (hi - lo)/2;

        if(mat[mid/m][mid%m] == target) return true;
        if(mat[mid/m][mid%m] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

2. // power function
--------------------
double myPow(double x, int y) {
    double ans = 1.0;
    long long n = y; // since (-INT_MIN) > INT_MAX that overflow int. 
    if(n < 0) n = -1*n;
    while(n) {
        if(n%2) {
            ans = ans * x;
            n--;
        } else {
            x = x*x;
            n = n/2;
        }
    }
    if(y<0) ans = (1.0)/(double)ans;
    return ans;
}

3. // Majority Element (>N/2 times) 
-----------------------------------
method 1: for each element traverse whole array and find ans. O(n^2)
mehtod 2: traverse once and use hashing
method 3: using Moore Voting Algorithm
int majorityElement(vector<int> &arr) {
    int count = 0;
    int candidate = 0;

    for(int ele: nums) {
        if(count == 0) candidate = ele;
        if(ele == candidate) count++;
        else count--;
    }
    return candidate;
}

4. // Majority Element II (>N/3 times)
--------------------------------------
method 1: brute
method 2: Boyer Moore Voting Algorithm
vector<int> majorityElement(vector<int> &nums) {
    int sz = nums.size();
    int num1 = -1, num2 = -1, count1 = 0, count2 = 0, i;
    for(i = 0; i < sz; i++) {
        if(nums[i] == num1) count1++;
        else if(nums[i] == num2) count2++;
        else if(count1 == 0) {
            num1 = nums[i];
            count1 = 1;
        } else if(count2 == 0) {
            nums2 = nums[i];
            count2 = 1;
        } else {
            count1--;
            count2--;
        }
    }
    vector<int> ans;
    count1 = count2 = 0;
    for(i = 0; i < sz; i++) {
        if(nums[i] == num1)
            count1++;
        else if(nums[i] == num2)
            count2++;
    }
    if(count1 > sz/3) ans.push_back(num1);
    if(count2 > sz/3) ans.push_back(num2);
    return ans;
}

5. // Grid Unique Paths
-----------------------
method 1: recursion. TC: O(2^n)
method 2: Memoization O(n*m)
int countPaths(int i, int j, int n, int m, vvi &dp) {
    if(i == (n-1) && j == (m-1)) return 1;
    if(i >= n || j >= m) return 0;
    if(dp[i][j] != -1) return dp[i][j];
    return dp[i][j] = countPaths(i+1, j, n, m, dp) + countPaths(i, j+1, n, m, dp);
}
method 3: Combination (m+n-2)C(m-1) or (m+n-2)C(n-1)

6. // Reverse Pairs (Leetcode)
------------------------------
method 1:
method 2: using mergeSort.
int merge(vector<int> &nums, int low, int mid, int high) {
    int cnt = 0;
    int j = mid + 1;
    for(int i = low; i <= mid; i++) { // main logic
        while(j <= high && nums[i] > 2LL * nums[j]) {
            j++;
        }
        cnt += (j - (mid+1));
    }

    // merging
    vector<int> temp;
    int left = low, right = mid+1;
    while(left <= mid && right <= high) {
        if(nums[left] <= nums[right]) {
            temp.push_back(nums[left++]);
        } else {
            temp.push_back(nums[right++]);
        }
    }

    while(left <= mid) temp.push_back(nums[left++]);
    while(right <= high) temp.push_back(nums[right++]);

    for(int i = low; i <= high; i++) {
        nums[i] = temp[i - low];
    }
    return cnt;
}

int mergeSort(vector<int> &nums, int low, int high) {
    if(low >= high) return 0;
    int mid = (low + high) / 2;
    int inv = mergeSort(nums, low, mid);
        inv += meggeSort(nums, mid + 1, high);
        inv += megre(nums, low, mid, high);
    return inv;
}

int reversePair(vector<int> &nums) {
    return mergeSort(nums, 0, nums.size()-1);
}


DAY 4: Hashing
==============
1. // 2 Sum problem
-------------------
method 1: two nested loops
method 2: use hashmap if target-arr[i] is not in map inset arr[i] to it. else we get the ans.
vector<int> twoSum(vector<int> arr, int t) {
    unordered_map<int, int> mp; // arr[i], i

    for(int i = 0; i < arr.size(); i++) {
        if(mp.find(t - arr[i]) == mp.end()) mp[arr[i]] = i;
        else return {mp[t - arr[i]], i};
    }
    return {-1, -1}; // pair doesn't exist
} 

2. // 4 Sum problem (unique quadruplates) 
-----------------------------------------
method 1: using sorting, 3 loops, and Binary Search
method 2: using sorting, 2 loops and two pointer
vector<vector<int>> fourSum(vector<int>& nums, int target) {
        
    vector<vector<int>> ans;
    
    sort(nums.begin(), nums.end());
    
    for(int i = 0; i < nums.size(); i++) {
        for(int j = i+1; j < nums.size(); j++) {
            int s = j+1, e = nums.size() - 1;
            long long newTar = target - nums[i] - nums[j];
            // applying two pointer to get newTarget in remaining right half of array
            
            while(s < e) {
                long long sum = nums[s] + nums[e];
                if(sum == newTar) {
                    ans.push_back({nums[i], nums[j], nums[s], nums[e]});
                    s++; e--;
                    // avoiding duplicates
                    while(s < e && nums[s-1] == nums[s]) s++;
                    while(s < e && nums[e+1] == nums[e]) e--;
                } else if(sum < newTar) s++;
                else e--;
            }
            
            // avoiding duplicates for j
            while(j+1 < nums.size() && nums[j+1] == nums[j]) j++;
        }
        while(i+1 < nums.size() && nums[i+1] == nums[i]) i++;
    }
    return ans;
}


3. // Longest Consecutive Sequence 
----------------------------------
method 1: sort and count;
method 2: using hashmap
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> s(nums.begin(), nums.end());
    int ans = 0;
    for(auto ele: nums) {
        if(s.find(ele - 1) == s.end()) { // means ele is minimum of a consicutive seq
            int currLen = 1, nextEle = ele + 1;
            
            while(s.find(nextEle) != s.end()) {
                currLen++; nextEle++;
            }
            ans = max(ans, currLen);
        }
    }
    return ans;
}
method 3: using hashmap 
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> s(begin(nums), end(nums)); // inserting all elements into hashset
    int longest = 0;
    for(auto& num : s) {
        int cur_longest = 1;
        // find consecutive elements in the backward and forward direction from num
        for(int j = 1; s.count(num - j); j++) s.erase(num - j), cur_longest++;
        for(int j = 1; s.count(num + j); j++) s.erase(num + j), cur_longest++;
        longest = max(longest, cur_longest);  // update longest to hold longest consecutive sequence till now
    }
    return longest;
}


4. // Largest Subarray with 0 sum
---------------------------------
method 1: two loops
method 2: use hashmap to keep track of prefix sum and index
int maxLen(int A[], int  n) {
    unordered_map<int, int> mp;
    int maxi = 0, sum = 0;
    for(int i = 0; i < n; i++) {
        sum += A[i];
        if(sum == 0) maxi = i;
        else {
            if(mp.find(sum) != mp.end()) maxi = max(maxi, i - mp[sum]);
            else mp[sum] = i;
        }
    }
    return maxi;
}

5. // Count number of subarrays with given XOR(this clears a lot of problems)
-----------------------------------------------------------------------------
int solve(vector<int> &A, int B) {
    unordered_map<int, int> freq;
    int cnt = 0, xorr = 0;
    for(auto it: A) {
        xorr = xorr ^ it;
        if(xorr == B) cnt++;
        if(freq.find(xorr ^ B) != freq.end()) cnt += freq[xorr ^ B];
        freq[xorr]++;
    }
    return cnt;
} 


6. // Longest substring without repeat character | Amazon
method 1: using set and two pointer.
method 2: uisng two pointer and hashing.
int lengthOfLongestSubstring(string s) {
    vector<int> mpp(256, -1); // hashtable

    int left = 0, right = 0; // moving both pointer front left to right
    int n = s.size();
    int len = 0;
    while(right < n) {
        // if char at left found before and lies between left and right update it
        if(mpp[s[right]] != -1) left = max(mpp[s[right]]+1, left);
        mpp[s[right]] = right; // new positin of char at right
        len = max(len, right - left + 1);
        right++;
    }
    return len;
}


DAY 5: Linked List
==================

1.// Reverse a LinkedList 
-------------------------

2.// Find middle of LinkedList 
------------------------------
method 1: count and then find middle (2 pass)
method 2: use tortoise method, using slow and fast pointer
3.// Merge two sorted Linked List 
----------------------------------
method 1: traverse both and make new node for each valid candidatae
method 2: use merge sort like technique

4.// Remove N-th node from back of LinkedList 
---------------------------------------------
method 1: count and find size - n th node. (2 pass)
method 2: use two pointer,assign head to a dummy, first move one to n, then move both till fast reaches to null

5.// Delete a given Node when a node is given. (0(1) solution) 
--------------------------------------------------------------
method 1: copy data and delete next;
    node->val = node->next->val;
    node->next = node->next->next;

6. // Add two numbers as LinkedList 
-----------------------------------

DAY 6:
======
1. // Find intersection point of Y LinkedList 
---------------------------------------------
method 1: traverse and get difference of both, then move a pointer on largest one till difference, then start moveing
          pointer on small one alse, check if both pointer are equal or not
method 2: push into stack and pop till address of top of stack are same.
method 3: one pass by swapping pointer
ListNode* getIntersectionNode(ListNode *h1, ListNode *h2) {
    if(h1 == NULL || h2 == NULL) return NULL;

    ListNode *a = h1, *b = h2;
    // if a & b have different len, then we will stop the loop after 2nd iteration
    while(a != b) {
        // for the end of first iteration, we just reset the pointer to the head of other linkdelist
        a = a==NULL? h2 : a->next;
        b = b==NULL? h1 : b->next;
    }
    return a;
}

2. // Detect a cycle in Linked List
-----------------------------------
method 1: use hashmap and check if address contains or not, if it contains there exist cycle else traverse till NULL.
method 1: floyed cycle algorithm (using fast and slow pointer)

3. // Reverse a LinkedList in groups of size k. 
-----------------------------------------------
ListNode* reverseKGroup(ListNode* head, int k) {
    if(head == NULL) return head;
    int n = k;
    ListNode *prev = NULL, *curr = head, *next;
    
    while(curr && n) {
        next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
        n--;
    }
    if(n) return reverseKGroup(prev, k-n); // if last group has less than k elements then, re reverse it.
    head->next = reverseKGroup(curr, k);
    return prev;
}

4. // Check if a LinkedList is palindrome or not. 
-------------------------------------------------
mehtod 1: count and find middle, push middle into stack ans match
method 2: find middle using slow fast and then reverse next after middle and then check head with slow

bool isPalindrome(ListNode head) {
    if(head == NULL || head->next == NULL)
        return true;
    ListNode *slow = head, *fast = head;

    while(fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
    }

    slow->next = reverse(slow->next);
    slow = slow->next;

    while(slow) {
        if(head->val != slow->val) return false;
        head = head->next;
        slow = slow->next;
    }
    return true;
}

5. // Find the starting point of the Loop of LinkedList
-------------------------------------------------------
method 1: using hashing
method 2: uisng slow and fast poiners (floyed cycle)
listNode detectCycle(listNode *head) {
    ListNode *slow = head, *fast = head;
    while(fast || fast->next) {
        slow = slow->next;
        fast = fast->next->next;

        if(slow == fast) {
            fast = head;
            while(slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    }
    return NULL;
}


6. // Flattening of a LinkedList 
--------------------------------
method 1: use recusion and mergesort for merging two linked list
Node *flatten(Node *head) {
    if(head == NULL || root->next == NULL) return root;
    // recur for list on right
    root -> next = flatten(root->next);
    // during backtraking
    return mergeTwoList(root, root->next); // on root->bttom, root->next->bottom
}

7. // Rotate a LinkedList
 ------------------------
 method 1: find last node and attach it to head, repeat it k times
 method 2: count & reach to last node make it circular LL and then move pointer 
           (size - k) more now set head as pointer->next & set pointer->next = NULL
ListNode *rotateRight(ListNode *head, int k) {
    // edge cases
    if(!head || !head->next || k==0) return head;

    ListNode *curr = head; int len = 1;
    while(curr->next && ++len) curr = curr->next;

    if(k%len == 0) return head;
    k = len - (k%len); // for (size - k) moves from head
    cur->next = head; // make LL as circular;

    while(k--) cur = cur->next;
    // make the node head and break connection
    head = curr->next;
    cur->next = NULL;

    return head;
}

DAY 7: Two Pointer
===================
1. // Clone a Linked List with random and next pointer
-----------------------------------------------------
method 1: using hashmap of <node*, node*> representing originalNode, deepCopy
Node* copyRandomList(Node* head) {
    if(head == NULL) return head;
    
    unordered_map<Node*, Node*> mp;
    
    Node *temp = head;
    
    while(temp) {
        mp[temp] = new Node(temp->val);
        temp = temp->next;
    }
    
    temp = head;
    Node *ans = mp[head];
    
    while(temp) {
        mp[temp]->next = mp[temp->next];
        mp[temp]->random = mp[temp->random];
        temp = temp->next;
    }
    
    return ans;
}
method 2: using pointers only, 1. insert deepCopy after earch node, 2. make random links, 3. move next pointers to its correct positon
Node* copyRandomList(Node* head) {
    if(head == NULL) return head;
    
    Node *curr = head;
    
    // 1. insetting copies of each node just after them
    while(curr) {
        Node *copy = new Node(curr->val);
        
        copy->next = curr->next; // new node points next node of original list
        curr->next = copy; // now curr is points to copy node.
        
        curr = curr->next->next; // moving curr to point next original node.
    }
    
    // 2. setting the random pointers or each   new copy nodes accordingly
    curr = head;
    while(curr) {
        curr->next->random = curr->random ? curr->random->next: NULL;
        curr = curr->next->next;
    }
    
    // 3. seperating both lits,  Only nedd to modify next pointers of each nodes
    Node *temp1 = head;
    curr = head->next;
    Node *dummy = new Node(0);
    Node *temp2 = dummy;
    int pos = 2; // odd will be old nodes and even will be new nodes
    
    while(curr) {
        if(pos&1) {
            temp1->next = curr;
            temp1 = temp1->next;
        } else {
            temp2->next = curr;
            temp2 = temp2->next;
        }
        curr = curr->next;
        pos++;
    }
    
    temp1->next = NULL; // always 2nd last node.
    
    return dummy->next;
}

2. // 3 sum 
------------
method 1: use three nested loops
method 2: a+b+c = 0 => (b+c) = 0 - a
vvi threeSum(vi &num) {
    sort(nums.begin(), nums.end());
    vvi res;

    //move for a
    for(int i = 0; i < nums.size(); i++) {
        if(i != 0 && (i > 0 && nums[i] == nums[i-1])) continue; // ignoring same value

        int lo = i+1, hi = nums.size()-1, sum = 0 - num[i]; // sum -> (b+c), b->lo, c->hi
        while(lo < hi) {
            if(nums[lo] + nums[hi] == sum) {
                res.push_back({nums[i], nums[lo], nums[hi]});

                while(lo < hi && nums[lo] == nums[lo+1]) lo++;
                while(lo < hi && nums[hi] == nums[hi-1]) hi++;
                lo++, hi--;
            }
            else if (nums[lo] + nums[hi] < sum) lo++;
            else hi--;
        }
    }
    return res;
}

3. // Trapping Rain Water
-----------------------
method 1: go to each index and look for max-left & max-right rain += min(maxL, maxR) - arr[i]; TC: O(n^2) SC: O(1)
method 2: precompute leftprefix & rightprefix arrays, rain += min(leftprefix[i], rightprefix[i]) - arr[i]; TC: O(3n) SC: O(2n)
method 3: using NGL & NGR (stack implementation) TC: O(3n) SC: O(2n), do not discuss in interview.
method 4: using Two Pointer

int trapWater(vector<int> &arr) {
    L = 0, R = n-1, maxLeft = arr[0], maxRight = arr[n-1];
    while(L < R) {
        if(arr[L] <= arr[R]) {
            if(arr[L] >= maxLeft) maxLeft = arr[L];
            else rain += maxLeft - arr[L];
            L++;
        } else {
            if(arr[R] >= maxRight) maxRight = arr[R];
            else rain += maxRight - arr[R];
            R++;
        }
    }
    return rain;
}

4. // Remove Duplicate from Sorted array 
----------------------------------------
method 1: use hashset
method 2: use two pointers

int removeDuplicates(vi &nums) {
    int i, j;
    for(i = 0, j = 0; j < nums.size(); j++) {
        if(nums[i] != nums[j]) {
            i++;
            nums[i] = nums[j];
        }
    }
    return i+1;
}

5. // Max consecutive ones
--------------------------  
int countMaxConsecutiveOnse(vi nums) {
    int cnt = 0, mx = 0;
    for(auto ele: nums) {
        if(ele == 0) cnt = 0;
        else cnt++;
        mx = max(mx, cnt);
    }
    return mx;
}


DAY 8: Greddy
==============
 1.// N meeting in one room 
 ---------------------------
 method 1: sort meeting according to their finish time
 struct meeting {
    int start, end, pos;
 };

 bool comparator(meeting a, meeting b) {
    if(a.end < b.end) return true;
    if(a.end > b.end) return false;
    if(a.pos < b.pos) return true;
    return false;
 }

 void maxMeetings(int s[], int e[], int n) {
    meeting meet[n];
    for(int i = 0; i < n; i++) {
        meet[i] = {s[i], e[i], i+1};
    }
    sort(meet, meet+n, comparator);

    int limit = meet[0].end;
    cout << meet[0].pos;

    for(int i = 1; i < n; i++) {
        if(meet[i].start > limit) {
            limit = meet[i].end;
            cout << meet[i].pos;
        }
    }
 }

 2.// Minimum number of platforms required for a railway
 -------------------------------------------------------
int findPlatForm(int arr[], int dep[], int n) {
    sort(arr, arr+n);
    sort(dep, dep+n);

    int plat_needed = 1, result = 1;
    int i = 1, j = 0;

    while(i < n && j < n) {
        if(arr[i] <= dep[j]) {
            plat_needed++;
            i++;
        } else if(arr[i] > dep[j]) {
            plat_needed--;
            j++;
        }
        if(plat_needed > result)
            result = plat_needed;
    }
    return result;
}


 3.// Job sequencing Problem 
 ---------------------------
 struct job { int dead, profit };
 
 pair<int, int> jobScheculing(job arr[], int n) {
    sort(arr, arr+n, [&](Job a, Job b){return a.profit > b.profit});
    int maxi = arr[0].dead;
    for(int i = 1; i < n; i++) maxi = max(maxi, arr[i].dead);

    vector<int> solt(maxi+1, -1);
    int countJobs = 0, jobProfit = 0;

    for(int i = 0; i < n; i++) {
        for(int j = arr[j].dead; j > 0; j--) {
            if(slot[j] == -1) {
                slot[j] = i;
                countJobs++;
                jonProfit += arr[i].profit;
                break;
            }
        }
    }
    return {countJobs, jobProfit};
 }


 4.// Fractional Knapsack Problem
 --------------------------------
 struct Item {int value, weight;};
 bool comp(Item a, Item b) {
    double r1 = (double)a.value / (double)a.weight;
    double r2 = (double)b.value / (double)b.weight;
 }

 double fractionalKnapsack(int W, Item arr[], int n) {
    sort(arr, arr+n, comp);

    int curWeight = 0;
    double finalvalue = 0.0;

    for(int i = 0; i < n; i++) {
        if(curWeight + arr[i].weight <= W) {
            curWeight += arr[i].weight;
            finalvalue += arr[i].value;
        } else { // will execute for last fractional part if space remains
            int remain = W - curWeight;
            finalvalue += (arr[i].value / (double)arr[i].weight) * (double) remain;
            break;
        }
    }
    return finalvalue;
 }

 5.// Greedy algorithm to find minimum number of coins
 -----------------------------------------------------
vector<int> findMin(int v) {
    int deno[] = {1, 2, 5, 10, 20, 50, 100, 500, 1000};
    int n = 9;
    vector<int> ans;
    for(int i = n - 1; i >= 0; i--) {
        while(v >= deno[i]) {
            v -= deno[i];
            ans.push_back(deno[i]);
        }
    }
    return ans;
}


 6.// Activity Selection (it is same as N meeting in one room) 
 -------------------------------------------------------------


DAY 9: Recursion
================
1. // Subset Sums
-----------------
method 1: power set
method 2: 
void func(int i, int sum, vector<int> &arr, vector<int> &sumSubset) { // input must be sorted
    if(i == arr.size()) {
        sumSubset.push_back(sum);
        return;
    }
    func(i+1, sum+arr[i], arr, sumSubset); // pick the element
    func(i+1, sum       , arr, sumSubset); // Do  not pick the element
}

2. // Subset-II 
---------------
void findSubsets(int ind, vector<int> &nums, vector<int> &ds, vector<vector<int>> ans) { // input must be sorted
    ans.push_back(ds);
    for(int i = ind; i < nums.size(); i++) {
        if(i != ind && nums[i] == nums[i-1]) continue;
        ds.push_back(nums[i]);
        findSubsets(i+1, nums, ds, ans);
        ds.pop_back();
    }
}

3. // Combination sum-1 
-----------------------
void findCombination(int ind, int target, vector<int> &arr, vector<vector<int>> &ans, vector<int> &ds) { // input must be sorted
    if(ind == arr.size()) {
        if(target == 0) ans.push_back(ds);
        return;
    }
    // pick up the element and stay
    if(arr[ind] <= target) {
        ds.push_back(arr[ind]);
        findCombination(ind, target - arr[ind], arr, ans, ds);
        ds.pop_back();
    }
    // not pick ans move
    findCombination(ind+1, target, arr, ans, ds);
}

4. // Combination sum-2 
-----------------------
void findCombination(int ind, int target, vector<int> &arr, vector<vector<int>> &ans, vector<int> &ds) { // input must be sorted
    if(target == 0) {
        ans.push_back(ds);
        return;
    }
    for(int i = ind; i < arr.size(); i++) {
        if(i>ind && arr[i] == arr[i-1]) continue;
        if(arr[i] > target) break;
        ds.push_back(arr[i]);
        findCombination(i+1, target - arr[i], arr, ans, ds);
        ds.pop_back();
    }
}

5. // Palindrome Partitioning 
-----------------------------
void func(int index, string s, vector<string> &ds, vector<vector<string> &res) {
    if(index == s.size()) {
        res.push_back(ds);
        return;
    }
    for(int i = index; i < s.size(); i++) {
        if(isPalindrome(s, index, i)) {
            ds.push_back(s.substr(index, i-index+1));
            func(i+1, s, ds, res);
            ds.pop_back();
        }
    }
}
bool isPalindrome(string str, int s, int e) {
    while(s <= e) 
        if(str[s++] != str[e--]) 
            return false;
    return true;
}

6. // K-th permutation Sequence
-------------------------------
string getPermutation(int n, int k) {
    int fact = 1;
    vector<int> numbers;
    for(int i = 1; i < n; i++) {
        fact = fact * i;
        numbers.push_back(i);
    }
    numberes.push_back(n);
    staring ans = "";
    k = k - 1; // kth permutation will be at k-1 th position in 0 index
    while(true) {
        ans = ans + to_string(numbers[k/fact]);
        numberes.erase(numberes.begin() + k / fact);
        if(numberes.size() == 0) break;
        k = k % fact;
        fact = fact / numberes.size();
    }
    return ans; 
}


DAY 10: BackTracking
====================
1. // Print all Permutations of a string/array
----------------------------------------------
void recurPermute(int index, vector<int> &nums, vector<vector<int>> &ans) {
    if(index == nums.size()) {
        ans.push_back(nums);
        return;
    }
    for(int i = index; i < nums.size(); i++) {
        swap(nums[index], nums[i]);
        recurPermute(index + 1, nums, ans);
        swap(nums[index], nums[i]);
    }
}

1. // N queens Problem 
----------------------
method 1: use while loop for checking (only three directionn is required to check)
method 2: use hashing to check saft condition to put a queen
        // sizes: leftRow(n), lowerDiagonal(2*n-1) and upperDiagonal(2*n-1)
void solve(int col, vector<string> &board, vector<vector<string>> &ans, vector<int> &leftRow,
        vector<int> &upperDiagonal, vector<int> &lowerDiagonal, int n) {
    if(col == n) {
        ans.push_back(board);
        return;
    }

    for(int row = 0; row < n; row++) {
        if(leftRow[row] == 0 && lowerDiagonal[row+col] == 0 && upperDiagonal[col-row + n-1]) {
            board[row][col] = 'Q';
            leftRow[row] = 1;
            lowerDiagonal[row+col] = 1;
            upperDiagonal[col-row + n-1] = 1;
            solve(col+1, board, ans, leftRow, upperDiagonal, lowerDiagonal, n);
            board[row][col] = '.';
            leftRow[row] = 0;
            lowerDiagonal[row+col] = 0;
            upperDiagonal[col-row + n-1] = 0;
        }
    }
}

1. // Sudoku Solver
-------------------
bool isValid(vector<vector<char>> &board, int row, int col, char c) {
    for(int i = 0; i < 9; i++) {
        if(board[i][col] == c) return false;
        if(board[row][i] == c) return false;
        if(board[3*(row/3) + i/3][3*(col/3) + i%3] == c) return false;
    }
    return true;
}
bool solve(vector<vector<char>> &board) {
    for(int i = 0; i < board.size(); i++) {
        for(int j = 0; j < board[0].size(); j++) {
            if(board[i][j] == '.') {
                for(char c = '1'; c <= '9'; c++) {
                    if(isValid(board, i, j, c)) {
                        board[i][j] = c;
                        if(solve(board) == true) return true;
                        else board[i][j] = '.';
                    }
                }
                return false; // if unable to fill with any one.
            }
        }
    }
    return true; // if all is filled;
}

1. // M coloring Problem 
------------------------
bool isSafe(int node, int color[], bool graph[101][101], int n, int col) {
    for(int i = 0; i < n; i++) {
        if(k != node && graph[k][node] == 1 && color[k] == col)
            return false;
    }
    return true;
}
bool solve(int node, int color[], int m, int N, bool graph[101][101]) {
    if(node == N) return true;

    for(int i = 1; i <= m; i++) {
        if(isSafe(node, color, graph, N, i)) {
            color[node] = i;
            if(solve(node+1, color, m, N, graph)) return true;
            color[node] = 0;
        }
    }
    return false;
}


1. // Rat in a Maze 
--------------------
void solve(int i, int j, vector<vector<int>> &a, int n, vector<string> &ans, string move, vector<vector<int>> &vis) {
    if(i == n-1 || j==n-1) {
        ans.push_back(move);
        return;
    }
    // downward
    if(i+1<n && !vis[i+1][j] && a[i+1][j] == 1) {
        vis[i][j] = 1;
        solve(i+1, j, a, n, ans, move + 'D', vis);
        vis[i][j] = 0;
    }

    // left
    if(j+1<n && !vis[i][j-1] && a[i][j-1] == 1) {
        vis[i][j] = 1;
        solve(i, j-1, a, n, ans, move + 'L', vis);
        vis[i][j] = 0;
    }
    
    // right
    if(j+1<n && !vis[i][j+1] && a[i][j+1] == 1) {
        vis[i][j] = 1;
        solve(i, j+1, a, n, ans, move + 'R', vis);
        vis[i][j] = 0;
    }
    
    // upward
    if(i-1<n && !vis[i-1][j] && a[i-1][j] == 1) {
        vis[i][j] = 1;
        solve(i-1, j, a, n, ans, move + 'D', vis);
        vis[i][j] = 0;
    }
    
}


Word Break (print all ways)  (Will be covered later in DP series)

DAY 11: Binary Search
======================
1. // Nth root of a number
--------------------------
double getNthRoot(int n, int m) { // nth root of m
    double low = 1, high = m, eps = 1e-6;

    while((high - low) > eps) {
        double mid = (low + high) / 2.0;
        if(multiply(mid, n) < m) { // multiply return mid*mid* ... times
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}

2. // Matrix Median
-------------------
method 1: store, sort and find the ans
method 2: binary Search
int countSmallerThanEqualToMid(vector<int> &row, int mid) {
    int l = 0, h = row.size() - 1;
    while(l <= h) {
        int md = (l+h) >> 1;
        if(row[md] <= mid) l = md + 1;
        else h = md - 1;
    }
    return l; // upper_bound, if looking for x, this will return next incex to x
}
int findMedian(vector<vector<int>> &A) {
    int l = 1, h = 1e9;
    int n = A.size(), m = A[0].size();
    while( l<= h ) {
        int mid = (l+h) >> 1;
        int cnt = 0; // count all vaues which is <= mid in matrix
        for(int i = 0; i<n; i++)
            cnt += countSmallerThanEqualToMid(A[i], mid);

        if(cnt <= (n*m)/2) // for median
            l = mid + 1;
        else h = mid - 1;
    }
    return l; // return index just 
}


3. // Find the element that appears once in sorted array, and rest element appears twice (Binary search) 
---------------------------------------------------------------------------------------------------------
method 1: using xor
method 2: binary Search based on index (i, i+1) always equal if number is not missing
int singleNonDuplicate(vector<int< &nums) {
    int low = 0, high = nums.size() - 2;
    while(low <= high) {
        int mid = (low + high) >> 1;
        if(nums[mid] == nums[mid^1]) //  if mid is odd mid^1 give next integer, else previous integer
            low = mid + 1;
        else
            high = mid - 1;
    }
    return nums[low];
}

4. // Search element in a sorted and rotated array/ find pivot where it is rotated 
----------------------------------------------------------------------------------
method 1: either left half is sorted or right half will be sorted *always. apply binary search on sorted part
int search(vector<int> &a, int target) {
    int l = 0, h = a.size()-1, mid;

    while(l <= h) {
        mid = (l+h) >> 1;

        if(nums[mid] == target) {
            return mid;
        } else if(nums[l] <= nums[mid]) { // if left half is sorted
            // applying binary search
            if(target >= a[l] && target < nums[mid]) h = mid - 1;
            else l = mid + 1;
        } else { // if right half is sorted
            // applying binary search
            if(target > a[mid] && target <= a[h]) l = mid + 1;
            else h = mid - 1;
        }
    }
}

5. // Median of 2 sorted arrays
--------------------------------
method 1: merge and sort then find middle one.
method 2: merge using merge sort technique and find middle one.
method 3: find middle in both sorted insted of merging using two pointer.
method 4: using binary Search:
double findMedianOfSortedArrays(vector<int> &arr1, vector<int> &arr2) {
    if(arr1.size() > arr2.size()) findMedianOfSortedArrays(arr2, arr1);
    int n1 = nums.size();
    int n2 = nums.size();
    int low = 0, high = n1;

    while(low <= high) {
        int cut1 = (low + high) >> 1;
        int cut2 = (n1 + n2 + 1)/2 - cut1;

        int l1 = cut1 == 0 ? INT_MIN : arr1[cut1 - 1];
        int l2 = cut2 == 0 ? INT_MIN : arr2[cut2 - 1];

        int r1 = cut1 == n1 ? INT_MAX : arr1[cut1];
        int r2 = cut2 == n2 ? INT_MAX : arr2[cut2];

        if(l1 <= r2 && l2 <= r1) {
            if((n1 + n2) % 2 == 0)
                return (max(l1, l2) + min(r1, r2)) / 2.0;
            else
                return max(l1, l2);
        } else if(l1 > r2) {
            high = cut1 - 1;
        } else {
            low = cut1 + 1;
        }
    }
    return 0.0;
}

6. // K-th element of two sorted arrays 
---------------------------------------
int kthElement(int arr1[], int arr2[], int n, int m, int k) {
    if(n > m) return kthElement(arr2, arr1, m, n, k);

    int low = max(0, k-m), high = min(k, n);

    while(low <= high) {
        int cut1 = (low+high) >> 1;
        int cut2 = k - cut1;
        int l1 = cut1 == 0 ? INT_MIN : arr1[cut1 - 1];
        int l2 = cut2 == 0 ? INT_MIN : arr2[cut2 - 2];
        int r1 = cut1 == n ? INT_MAX : arr1[cut1];
        int r2 = cut2 == m ? INT_MAX : arr2[cut2];

        if(l1 <= r2 && l2 <= r2) {
            return max(l1, l2);
        } else if(l1 > r2) {
            high = cut1 - 1;
        } else {
            low = cut1 + 1;
        }
    }
    return -1;
}

7. // Allocate Minimum Number of Pages
--------------------------------------
method 1: Binary Search
bool check(int A[], int barrier, int M, int N) {
    int currStudent = 1, currPages = 0;
    
    for(int i = 0; i < N; i++) {
        if(A[i] > barrier) return false;
        
        if(A[i] + currPages > barrier) {
            currStudent++;
            currPages = A[i];
        } else {
            currPages += A[i];
        }
    }
    
    if(currStudent > M) return false;
    return true;
}

public:
int findPages(int A[], int N, int M) {
    int hi = accumulate(A, N);
    int lo = maxElement(A, N);
    int ans = INT_MAX;
    
    while(lo <= hi) {
        int mid = (lo + hi) >> 1;
        
        if(check(A, mid, M, N)) {
            hi = mid - 1;
            ans = min(mid, ans);
        }
        else lo = mid + 1;
    }
    return ans == INT_MAX ? -1 : ans;
}

8. // Aggressive Cows (SPOJ)
---------------------
method 1: same as allocate books

DAY 12: Bits
============
DAY 13: Stack and Queues
========================
1. // Implement Stack Using Arrays 
----------------------------------
2. // Implement Queue Using Arrays
----------------------------------
3. // Implement Stack using Queue (using single queue)
------------------------------------------------------
method 1: use tow queues. q1, q2.
        // for push operation
        1) push into q2. // only one element in q2
        2) push all elements of q1 in it. // q1 become empty after this
        3) swap(q1, q2); // this time always q2 will empty
method 2: use only one queue.
        // for push operation
        1) push in Q
        2) pop & push Q.size()-1 elements of same queue again. // rotating

4. // Implement Queue using Stack (O(1) amortised method)
---------------------------------------------------------
method 1: using couple of stakc s1, s2.
        // for push ooperation
        1) pop all from s1 to s2.
        2) push element to s1.
        3) pop all form s2 to s1.
method 2: using couple of stack 'input', 'output' with amoritised O(1) TC
        // for push Operation 
        add element to input;
        
        // for pop()
        if(output not empty) output.pop();
        else {
            pop all input and push into output;
            output.pop();
        }

        // for top()
        if(output not empty) return output top
        else {
            pop all input and push to output;
            return output.top();
        } 


5. // Check for balanced parentheses
------------------------------------

6. // Next Greater Element
--------------------------
vector<int> nextGreaterElement(vector<int> nums) {
    vector<int> nge(n);
    stack<int> stk;
    for(int i = n-1; i >= 0; i--) {
        while(stk.size() && stk.top() <= nums[i]) stk.pop();

        if(st.size()) nge[i] = stk.top();
        else nge[i] = -1;

        stk.push(nums[i]);
    }
}

7. // Sort a Stack
------------------
method 1: using extra space.
method 2: usng recursion O(1) space.
void insertAtCorrectPosition(stack<int> &s, int val) {
    if(s.size() == 0 or s.top() < val) {
        s.push(val);
        return;
    }
    
    int temp = s.top(); s.pop();
    insertAtCorrectPosition(s, val);
    s.push(temp);
}

void sortStack(stack<int> &s) {
    if(s.size() == 0) return;
    int temp = s.top(); s.pop();
    sortStack(s);
    insertAtCorrectPosition(s, temp);
}

DAY 14: Stack and Queues
========================
1. // LRU cache (vvvv. imp)
--------------------------- 


2. // LFU Cache (Hard, can be ignored)
-------------------------------------- 


3. Largest rectangle in histogram (Do the one pass solution)
------------------------------------------------------------
Two pass: 
One pass: 


4. // Sliding Window maximum
----------------------------


5. // Implement Min Stack
-------------------------


6. // Rotten Orange (Using BFS)
-------------------------------


7. // Stock Span Problem
------------------------


8. // Find maximum of minimums of every window size
---------------------------------------------------


9. // The Celebrity Problem
---------------------------




DAY 15:
DAY 16:
DAY 17:
DAY 18:
DAY 19:
DAY 20:
DAY 21:
DAY 22:
DAY 23:
DAY 24:
DAY 25:
DAY 26:
DAY 27:
DAY 28:
DAY 29:
DAY 30:

// MORE

// Longest Palindromic Subsequence LPS.

void LPS(int i, int j, string &s) {
    if(i == j) return 1; // pointing to same char

    if(i > j) return 0;  // recursion is over
    if(s[i] == s[j]) return 2 + LPS(i+1, j-1, s); // match

    return max(LPS(i+1, j, s), LPS(i, j-1, s)); // no match
}


// power set algorithm (Petr and Combination Lock)

a = [2, 3, 4]

for(int i = 0; i < (1 << n)-1; i++) { // till 2^n - 1, all these integers have diff bit combinations, take advantage of that
    for(int bit = 0; bit < n; bit++) {
        if(i & (1 << bit)) {
            set.pb(a[bit]);
            // do somthing (can be used to make set if constraint is around 18)
        } else {
            // do somthing
        }
    }
}


rain water trapping: leftprefix, rightprefix