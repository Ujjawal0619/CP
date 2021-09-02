// QUICK SORT
int partation(vector<int> &arr, int s, int e) { // 0, size - 1
    int pivot = arr[e]; // selection last element as pivot;
    int j = s; 
    
    for(int i = s; i <= e-1; i++) 
        if(arr[i] < pivot) { // if we get smaller element
            swap(arr[i], arr[j]);
            j++;
        }

    swap(arr[j], arr[e]); // placint pivot to its correct place;
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
    