1. Reverse a string without affecting special characters
2. 2nd most frequent character in a string. 1st also
3. implement linked list and reverse
4. cycle in linked list
5. string rev, word reverse
6. binary search
7. left view of binary tree
8. product of two value with max number
9. leaders in array
10. anagram
11. compount interest for n year   double A = principle * (pow((1 + rate / 100), time));
12. LCA to binary and binary search tree
14. count all subarray of prime sum
13. sucessor and predecessor in BST
15. find pair whose sum is equal to K in array
16. distance between 2 given nodes
17. merge sort
18. Dikastras algo
19. stack using array, linked list, queue
20. unique value in array
21. count of repeated value in array
22. LRU cache (variation)
23. add two linked list
24. LPS (sub string)
25. Min Stack
26. Trie Implementation
27. binary tree form level, inorder
28. Min step to reach end




// LCA in Binary tree
struct Node *findLCA(struct Node* root, int n1, int n2)
{
    // Base case
    if (root == NULL) return NULL;
 
    // If either n1 or n2 matches with root's key, report
    // the presence by returning root (Note that if a key is
    // ancestor of other, then the ancestor key becomes LCA
    if (root->key == n1 || root->key == n2)
        return root;
 
    // Look for keys in left and right subtrees
    Node *left_lca  = findLCA(root->left, n1, n2);
    Node *right_lca = findLCA(root->right, n1, n2);
 
    // If both of the above calls return Non-NULL, then one key
    // is present in once subtree and other is present in other,
    // So this node is the LCA
    if (left_lca && right_lca)  return root;
 
    // Otherwise check if left subtree or right subtree is LCA
    return (left_lca != NULL)? left_lca: right_lca;
}

// LCA of BST
node *lca(node* root, int n1, int n2) {

    if (root == NULL) return NULL;
 
    // If both n1 and n2 are smaller
    // than root, then LCA lies in left
    if (root->data > n1 && root->data > n2)
        return lca(root->left, n1, n2);
 
    // If both n1 and n2 are greater than
    // root, then LCA lies in right
    if (root->data < n1 && root->data < n2)
        return lca(root->right, n1, n2);
 
    return root;
}


// Left view / Right view
void leftViewUtil(Node *root, int level, int &max_level) {

    if (root == NULL) return;
 
    if (max_level < level) {
        cout << root->data << " ";
        max_level = level;
    }

    leftViewUtil(root->left, level + 1, max_level); // recur right first for right view
    leftViewUtil(root->right, level + 1, max_level);
}
void printLeftView(Node* root) {
    if (!root) return;
 
    queue<Node*> q;
    q.push(root);
 
    while (q.size()) {    
        int n = q.size(); // number of nodes at current level
         
        for(int i = 1; i <= n; i++) { // Traverse all nodes of current level
            Node* temp = q.front(); q.pop();

            if (i == 1) cout<<temp->data<<" "; // use i == n for right view
             
            if (temp->left) q.push(temp->left);
            if (temp->right) q.push(temp->right);
        }
    }
}

// boundary traversal
void getLeftView(Node *root, vector<int> &ans) {
    if(root == NULL) return;
    
    if(root->left) {
        ans.push_back(root->data);
        getLeftView(root->left, ans);
    }
    else if(root->right) { // note 'else if' for one direction only
        ans.push_back(root->data);
        getLeftView(root->right, ans);
    }
}

void getBottomView(Node *root, vector<int> &ans) {
    if(root == NULL) return;
    
    if(root->left == root->right) { // leaf node
        ans.push_back(root->data);
    }
    
    getBottomView(root->left, ans);
    getBottomView(root->right, ans);
}

void getRightView(Node *root, vector<int> &ans) {
    if(root == NULL) return;
    
    if(root->right) {
        getRightView(root->right, ans);
        ans.push_back(root->data);
    }
    else if(root->left) {
        getRightView(root->left, ans);
        ans.push_back(root->data);
    }
}


vector <int> printBoundary(Node *root) {
    vector<int> ans;
    if(root == NULL) return ans;

    ans.push_back(root->data); // root node
    getLeftView(root->left, ans);
    getBottomView(root, ans);
    getRightView(root->right, ans);

    return ans;
}

// Spiral / Zigzag traversal
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    if(root == NULL) return {};
    
    vector<vector<int>> ans;
    
    stack<TreeNode*> lr, rl;
    
    rl.push(root);
    TreeNode *t;
    
    while(lr.size() || rl.size()) {
        vector<int> v;
        if(rl.size())
            while(rl.size()) {
                t = rl.top(); rl.pop();
                v.push_back(t->val);
                if(t->left) lr.push(t->left); // insert right first in stk 2
                if(t->right) lr.push(t->right);
            }
        else
            while(lr.size()){
                t = lr.top(); lr.pop();
                v.push_back(t->val);
                if(t->right) rl.push(t->right);
                if(t->left) rl.push(t->left);
            }
        ans.push_back(v);
    }
    return ans;
}

// inorder sucessor and predessor in bst.
if (root->data == key)
for suce (baad): go one right, go extream left;
for prec(pahle): go one left, go extream right;

if(key < root->val)
suce = root, call to left;

if(key > root->val)
pred = root, call to right;
void findPreSuc(Node* root, Node*& pre, Node*& suc, int key)
{
    if(root == NULL) return;
    
    if(root->key == key) {
        if(root->left) {
            Node* temp = root->left;
            while(temp->right)
                temp = temp->right;
            pre = temp;
        }
        
        if(root->right) {
            Node *temp = root->right;
            while(temp->left)
                temp = temp->left;
            suc = temp;
        }
    } else if(root->key > key) {
        suc = root;
        findPreSuc(root->left, pre, suc, key);
    } else {
        pre = root;
        findPreSuc(root->right, pre, suc, key);
    }
}


// Trie Data Structure
struct TrieNode {
    TrieNode* children[26];
    bool isLeaf;
    TrieNode() {
        isLeaf = false;

        for(int i = 0; i < 26; i++)
            children[i] = NULL;
    }
};


void insert(TrieNode *root, string key) {
    TrieNode *p = root;

    for(int i = 0; i < key.size(); i++) {
        int index = key[i] - 'a';

        if(p->children[index] == NULL)
            p->children[index] = new TrieNode();
        p = p->children[index];
    }
    
    p->isLeaf = true;
}

bool search(TrieNode *root, string key) {
    TrieNode *p = root;

    for(int i = 0; i < key.size(); i++) {
        int index = key[i] - 'a';

        if(p->children[index] == NULL)
            return false;

        p = p->children[index];
    }

    return p->isLeaf;
}

