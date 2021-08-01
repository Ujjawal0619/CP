1. Sort an array of 0’s 1’s 2’s without using extra space or sorting algo

// Durch National Flag Algorithm
void sortColors2(vector<int>& a) {
    int low = 0, mid = 0, hi = a.size() - 1;
    
    while(mid <= hi) {
        switch(a[mid]) {
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

2. Repeat and Missing Number

vector<int> findErrorNums(vector<int>& nums) {
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
    
    // if twice is xor0
    for (int i = 0; i < nums.size(); i++)
        if (nums[i] == xor0)
            return {xor0, xor1};
    
    return {xor1, xor0};

    
    // METHOD 1:
    long long n = nums.size();
    
    long long sum = n*(n + 1)/2; // X
    long long sqrsum = n*(n + 1)*(2*n + 1)/6; // X^2
    
    for(int ele: nums) {
        sum -= ele; // Y
        sqrsum -= pow(ele, 2); // Y^2
    }
   
    int t = sqrsum / sum; // a+b
    int a = (t- sum)/2;
    int b = (t + sum)/2;
    
    return {a, b};

    // METHOD 3: using frequency array, missing will be 0 and repting will be 2 count.
}



DAY 7

3. Trapping Rain Water
-----------------------
method 1: go to each index and look for max-left & max-right rain += min(maxL, maxR) - arr[i]; TC: O(n^2) SC: O(1)
method 2: precompute leftprefix & rightprefix arrays, rain += min(leftprefix[i], rightprefix[i]) - arr[i]; TC: O(3n) SC: O(2n)
method 3: using NGL & NGR (stack implementation) TC: O(3n) SC: O(2n), do not discuss in interview.
method 4: using Two Pointer

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