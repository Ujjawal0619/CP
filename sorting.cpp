// QUICK SORT
int partation(vector<int> &arr, int s, int e) { // 0, size - 1
    int pivot = arr[e]; // selection last element as pivot;
    int j = s; 
    
    for(int i = s; i <= e-1; i++) 
        if(arr[i] < pivot) { // if we get smaller element
            swap(arr[i], arr[j]);
            j++;
        }

    swap(arr[j], arr[e]); // place pivot to its correct position;
    return j;
}

void quickSort(vector<int> &arr, int s, int e) {
    if(s < e) {
        int i = partation(arr, s, e);
        
        quickSort(arr, s, i-1);
        quickSort(arr, i+1, e);
    }
}

// MERGE SORT
void merge(vector<int>& arr, int s, int m, int e) {
    int i = s, j = m+1;
    
    vector<int> buffer;
    
    while(i <=m && j <=e) {
        if(arr[i] <= arr[j]) buffer.push_back(arr[i++]);
        else buffer.push_back(arr[j++]);
    }
    
    while(i <= m) buffer.push_back(arr[i++]);
    while(j <= e) buffer.push_back(arr[j++]);
    
    for(int k = s; k <= e; k++) {
        arr[k] = buffer[k - s];
    }
}

void mergeSort(vector<int> &arr, int s, int e) {
    if(s < e) {
        int mid = (s+e)/2;
        
        mergeSort(arr, s, mid);
        mergeSort(arr, mid+1, e);
        
        merge(arr, s, mid, e);
    }
}

// SELECTION SORT
vector<int> sortArray(vector<int>& nums) {

    for (i = 0; i < nums.size(); i++)
        for (j = i+1; j < nums.size(); j++)
            if (nums[j] < nums[i])
                swap(nums[j], nums[i]);
    return nums;
}


// BUBBLE SORT
void bubbleSort(vector<int> &arr) {
    bool flag;

    for(int i = 0; i < n-1; i++) {
        flag = false;
        for(int j = 0; j < n-i-1; j++) {
            if(arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
                flag = true;
            }
        }
        if(flag == true)
            break;
    }
}


// INSERTION SORT
void insertionSort(vector<int> &arr) {
    int j, key;

    for(int i=1; i < n; i++) {
        key = arr[i];
        j = i-1;

        while(j>=0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}


// MIN/MAX HEAP
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