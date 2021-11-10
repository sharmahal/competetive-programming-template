bool cmp(pair<string, int>& a,
         pair<string, int>& b)
{
    return a.second < b.second;
}

int gcd(int a, int b){
if (a==0){
    return b;
}
return gcd(b%a,a);


}

int lcm(int m, int n) {
	return a*b/(__gcd(a,b));
}


int factorial(int n)
{
    // single line to find factorial
    return (n==1 || n==0) ? 1: (n * factorial(n - 1)%mod);
}

// modular inverse
int md=998244353;
inline int inv(int a) {
 int res=1;
 int n=md-2;
 while (n){
    if (n%2) res=(res*a)%md;
    n/=2;
    a=(a*a)%md;
 }
 return res;
}

string add(string a,string b){
	if(a.size()<b.size())swap(a,b);
	string ans;
	int n=a.size();
	int n1=a.size()-b.size();
	string temp;
	for(int i=0;i<n1;i++)temp+="0";
	b=temp+b;
	int car=0;
	for(int i=n-1;i>=0;i--){
		ans+=(char)((a[i]-'0'+b[i]-'0'+car)%10+'0');
		car=(a[i]-'0'+b[i]-'0'+car)/10;
	}
	if(car)ans+=(char)(car+'0');
	reverse(all(ans));
	return ans;
}

bool cmp(string s1,string s2){
	if(s1.size()!=s2.size())return s1.size()<s2.size();
	int n=s1.size();
	for(int i=0;i<n;i++)
		if(s1[i]>s2[i])return 0;
		else if(s1[i]<s2[i])return 1;
	return 1;
}


vector<int> z_function(string s) {
    int n = (int) s.length();
    vector<int> z(n);
    for (int i = 1, l = 0, r = 0; i < n; ++i) {
        if (i <= r)
            z[i] = min (r - i + 1, z[i - l]);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]])
            ++z[i];
        if (i + z[i] - 1 > r)
            l = i, r = i + z[i] - 1;
    }
    return z;
}


//binary exponentiation 
int modp(int a, int b, int mod)
{
    int ans=1;
    while(b)
    {
        if(b&1)
            ans=(ans*a)%mod;
        b/=2;
        a=(a*a)%mod
;    }
    return ans;
}


void reverseStr(string& str)
{
    int n = str.length();

    // Swap character starting from two
    // corners
    for (int i = 0; i < n / 2; i++)
        swap(str[i], str[n - i - 1]);
}

void bfs(vector<set<int>> g, int s, int dist[])
{
	bool mark[1010]; memset(mark, 0, sizeof mark);
	dist[s] = 0;
	queue<int> q;
	q.push(s);
	mark[s] = 1;
	while (!q.empty())
	{
		int u = q.front();
		q.pop();
		for (auto v : g[u])
		{
			if (mark[v]) continue;
			dist[v] = 1 + dist[u];
			mark[v] = 1;
			q.push(v);
		}
	}
}



vector<pair<int,int>> wg[100050];

int f=0;
//vi d(100050,INF);
 
vi dijkstra(int src)
{
    set<pair<int,int>> s;

    // vi parent(n+1,-1);
    
    d[src]=0;
    s.insert({0,src});
    while(!s.empty()){
        int i=s.begin()->second;
        s.erase(s.begin());
        for (auto & e:wg[i]){
            int nb=e.first;
            int w=e.second;
            if (d[i]+w<d[nb]){
                s.erase({d[nb],nb});
                d[nb]=d[i]+w;
                s.insert({d[nb],nb});
                // parent[nb]=i;
            }
        }
    }
   // return parent;
    return d;

}

int lim=150005;
int sz;
vector<int> adj[150005];
vi vis(lim,0);

void dfs(int x){
if (vis[x]==1)
    return;

sz++;
vis[x]=1;

for (auto &it:adj[x]){
    if (vis[it]!=1){
        dfs(it);
    }
}
}

const int N=1e5+5;
vector<int>prime;
int primes[N+1]={0};
void prime_sieve()
{ for(int i=3; i<=N; i+=2)
    {
        primes[i]=1;
    }
    for(int i=3; i<=N; i++){
        if(primes[i]==1)
        {
        for(int j=i*i; j<=N; j+=i)
        { 
           primes[j]=0;
        }
    }
 
    }
    primes[2]=1;
 
}

prime_sieve();
for(int i=0; i<=N; i++)
{
   if(primes[i]){
            prime.pb(i);
     for (int j=i;j<100002;j+=i){
        //spf for prime factorisation 
        if (spf[j]==0) spf[j]=i; //smallest prime factor (factor=spf[x]; x=x/spf[x] do this to get all prime factor)
     }
 
     }
 }

vector<pair<int,int>> primefact(int x){
    vector<pair<int,int>> p;
    int t=x;
    while (prime[j]*prime[j]<=a[i]){
            int c=0;
        while (t%prime[j]==0){
            t/=prime[j];
            c++;
            if (t==1){
                break;
            }
        }
         if (c){
           p.pb(prime[j],c);
         }
         j++;
        }
        
      if(t!=1){
      	p.pb({t,1})
      }
    return p;
}


void primefact(int n){
	for (int i=2;i*i<=n;i++){
    if (n%i==0){
    pf.pb(i);
    while(n%i==0)
        n/=i;
}
}

//prime fact
for (int i=0;prime[i]*prime[i]<=a;i++){
        while((a%prime[i])==0){
            cout<<a<<" "<<prime[i]<<endl;
            div.insert(prime[i]);
            a=a/prime[i];
        }
    }

    if (a!=1) div.insert(a); //the prime that is bigger than root a is inserted here 



////////////////////////////////////////////////////////////////////////
int C[1002][1002]; 
void binomialCoeff(int n, int k) 
{ 
    //nCk
    
    int i, j; 
  
    // Caculate value of Binomial Coefficient 
    // in bottom up manner 
    for (i = 0; i <= n; i++) 
    { 
        for (j = 0; j <= min(i, k); j++) 
        { 
            // Base Cases 
            if (j == 0 || j == i) 
                C[i][j] = 1; 
  
            // Calculate value using previously 
            // stored values 
            else
                C[i][j] = (C[i - 1][j - 1]%mod+ 
                          C[i - 1][j]%mod)%mod; 
        } 
    } 
  
    
} 

////////////////////////////////////////////////////////////////////////////


//calculating binomial coeffiecnt using 1d array (bigger numbers)
const int MAXN = 4e5;
int factorial[MAXN];
void pre(){
    factorial[0] = 1;
    for (int i = 1; i < MAXN; i++){
        factorial[i] = (factorial[i - 1] * i) % mod;
    }
}
int inverse(int a, int m = mod){
    int m0 = m;
    int y = 0, x = 1;
 
    if (m == 1)
        return 0;
 
    while (a > 1) {
        // q is quotient
        int q = a / m;
        int t = m;
 
        // m is remainder now, process same as
        // Euclid's algo
        m = a % m, a = t;
        t = y;
 
        // Update y and x
        y = x - q * y;
        x = t;
    }
 
    // Make x positive
    if (x < 0)
        x += m0;
 
    return x;
}
int bc(int n, int k) {
    if (n<k) return 0;
    return factorial[n] * inverse(factorial[k] * factorial[n - k] % mod) % mod;
}

/////////////////////////////////////////////////////////////////////////
bool palindrome (int a){
    int check[50]; int last=0;
    while (a){
        check[last++]=a%10;
        a=a/10;
    }
    for (int i=0;i<last;i++) {
        if (check[i]!=check[last-i-1]){
            return false;
        }
    } 
    return true;
}


// finding cycles (non intersecting) each vertex has one edge
int lim=250005;
int sz;
vector<int> adj[150005];
vi vis(lim,0);
vector<vi> cycles;
vi path;

void dfs(int x){
path.pb(x);
vis[x]=1;
int to=adj[x][0];
if (vis[to]==1){

    vi k;
    int i=path.size()-1;
    while(path[i]!=to){
        k.pb(path[i--]);
    }
    k.pb(to);
    cycles.pb(k);
}
else if (vis[to]==0){
    dfs(to);
}
path.pop_back();
vis[x]=2;


}

////////////////////////////////////////////////////////////////////
// dsu 

(most efficient one out of all below)
int find(vector<int>&parent, int i)
{
         return (i==parent[i])?i:parent[i]=find(parent,parent[i]);
}

void unionDSU(vector<int>&parent,vector<int>&rank, int i, int j)
{
         i = find(parent,i);
         j=find(parent,j);
         if (rank[i]>rank[j])
         {
             parent[j]=i;
             rank[i]++;
         }
         else 
         {
             parent[i]=j;
             rank[j]++;
         }
}


void make_set(int v) {
    parent[v] = v;
}

int find_set(int v) {
    if (v == parent[v])
        return v;
    return find_set(parent[v]);
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b)
        parent[b] = a;
}

// use rank
 void union_(pair<int,int>a ,pair<int,int>b)
    {
        a=find(a);
        b=find(b);
        if(rank[a.first][a.second]>rank[b.first][b.second])
        {
            par[b.first][b.second]=a;
            rank[a.first][a.second]++;
        }
        else
        {
            par[a.first][a.second]=b;
            rank[b.first][b.second]++;
        }
    }
////////////////////////////////////////////////////////
// to know the size of connected component 
vi parent (1005);
vi si(1005);

int find_set(int v) {
    if (v == parent[v])
        return v;
    return find_set(parent[v]);
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b)
       {
        if (si[a]<si[b]){
            swap (a,b);
        }
        parent[b]=a;
        si[a]+=si[b];
       }
}


//Strongly Connected Component 
vi g[1001];
bool vis[1001] , onStack[1001];
int in[1001] , low[1001];
stack<int> st;
 
int timer = 1 , SCC = 0;
 
void dfs_tarjan(int node)
{
    vis[node] = 1;
    in[node] = low[node] = timer++;
    onStack[node] = true;
    st.push(node);
 
    for(int u : g[node])
    {
        if((vis[u] == true) && (onStack[u] == true))
        {
            low[node] = min(low[node] , in[u]);
        }
        else
        if(vis[u] == false)
        {
            dfs_tarjan(u);
 
            if(onStack[u] == true)
            low[node] = min(low[node] , low[u]);
        }
    }
 
    if(in[node] == low[node])
    {
        SCC++;
        cout<<"SCC #"<<SCC<<endl;
 
        int u;
 
        while(1)
        {
            u = st.top();
            st.pop() , onStack[u] = false;
            cout<<u<<" ";
 
            if(u == node) break;
        }
        cout<<endl;
    }
}
////////////////////////////////////////////////////////////////////////////////////

//fenwick tree 
void update(int i, int val){
    while (i<=n){
        //max freq=n
        fen[i]+=val;
        i+=i&(-i);
    }
}

int sum(int i){
    int ans=0;
    while (i>0){
        ans+=fen[i];
        i-=i&(-i);
    }
    return ans;
}

//finding max 
void update(int row , int idx , int val){
    //val is real val, not increment
    while(idx < N){
        bit[row][idx] = max(bit[row][idx] , val);
        idx += idx & -idx;
    }
}
int query(int row , int idx){
    int res = 0;
    while(idx > 0){
        res = max(res , bit[row][idx]);
        idx -= idx & -idx;
    }
    return res;
}

/////////////////////////////////////////////////////////////////////////////////
//lca 
vi adj[200005];
int x,k;
int par[200005][22];
int itime[200005],otime[200005];
int t=0;

void dfs(int x,int p){
if (p!=-1){
    par[x][0]=p;
}
else par[x][0]=p;

itime[x]=t++;
for (int i=1;i<22;i++){
    if (par[x][i-1]==-1) par[x][i]=-1;
    else par[x][i]=par[par[x][i-1]][i-1];
}
for (auto it :adj[x]){
    if (it!=p) dfs(it,x);
}
otime[x]=t++;
}

int is_ancestor(int a,int b){
    return (itime[a]<=itime[b] && otime[a]>=otime[b]);
}

int lca(int a,int b){
    if (is_ancestor(a,b)) return a;
    if (is_ancestor(b,a)) return b;
    for (int i=21;i>=0;i--){
        //if (par[a][i]==-1) continue;
        if (par[a][i]!=-1 && !is_ancestor(par[a][i],b)){
            a=par[a][i];
        }

    }
    return par[a][0];
}
///////////////////////////////////////////////////////////////////////////////
//segment tree to get & from l to r v=1, r is one bigger than range calculating for 

const int N = 1000 * 1000;
int MAXBIT=30;
int a[N], t[4 * N];
int sum[N];
 
 
inline void build(int v, int l, int r) {
    if (l + 1 == r) {
        t[v] = a[l];
        return;
    }
    int mid = (l + r) >> 1;
    build(v * 2, l, mid);
    build(v * 2 + 1, mid, r);
    t[v] = t[v * 2] & t[v * 2 + 1];
}
 
inline int query(int v, int l, int r, int L, int R) {
    if (l == L && r == R) {
        return t[v];
    }
    int mid = (L + R) >> 1;
    int ans = (1ll << MAXBIT) - 1;
    if (l < mid) 
        ans &= query(v * 2, l, std::min(r, mid), L, mid);
    if (mid < r) 
        ans &= query(v * 2 + 1, std::max(l, mid), r, mid, R);
    return ans;
}
///////////////////////////////////////////////////////////////////////////////
binary trie 
struct node{
    int val;
    int cnt;
    node* child[2];
};
 
node* get_node()
{
    node* np = new node;
    np->child[0] = np->child[1] = NULL;
    np->val = 1;
    np->cnt = 0;
    return np;
}
 
void insert(int n  , node* root)
{
    node* tmp = root;
    for(int i=30;i>=0;i--)
    {
        bool byt = n & (1<<i);
        if(tmp->child[byt]==NULL)
        tmp->child[byt] = get_node();
        tmp = tmp->child[byt];
        tmp->cnt++;
    }
    tmp->val = n;
}
 
void delNode(int n , node* root)
{
    node* tmp = root;
    for(int i=30;i>=0;i--)
    {
        bool byt = n & (1<<i);
        tmp = tmp->child[byt];
        tmp->cnt--;
    }
}

//get minimum xor
int getmin(int n, node* root){
    node* tmpr=root;
    for (int bit=30;bit>=0;bit--){
        bool need=(n&(1<<bit));
        if (tmpr->child[need]==NULL){
            tmpr=tmpr->child[!need];
        }
        else {
            if (tmpr->child[need]->cnt==0){
                tmpr=tmpr->child[!need];
            }
            else {
                tmpr=tmpr->child[need];
            }
        }
    }
    delNode(tmpr->val,root);
    return (tmpr->val^n);

}
////////////////////////////////////////////////////////////////////////////
trie for words 
struct trie
{
    
    trie* child[26];
    string word;
};

trie* create()
{
    
    trie* temp=new trie();
    temp->word="";
    // memset(temp->child,NULL,sizeof(temp->child));
    return temp;
}

void add(trie* root,string s)
{
    for (int i=0;i<s.length();i++)
    {
        if(root->child[s[i]-'a']==NULL)
            root->child[s[i]-'a']=create();
        root=root->child[s[i]-'a'];
    }
    
    root->word=s;
}
trie* buildTrie(vector<string>& words) {
        trie* root = create();
        for (string w : words) {
            add(root,w);
        }
        return root;
    }
//here we are checking the words that are in Trie and in the matrix, this is the find function here 
//Normal word find can also happen where we keep going untill we find the child is not NULL  
void dfs (vector<vector<char>>& board, int i,int j,trie* p, vector<string> &res){
        char c=board[i][j];
        if (visit[i][j] || p->child[c-'a']==NULL) return;
        p=p->child[c-'a'];
        if (p->word!=""){
            res.push_back(p->word);
            p->word="";
        }
        visit[i][j]=1;
         if (i > 0) dfs(board, i - 1, j ,p, res);
        if (j > 0) dfs(board, i, j - 1, p, res);
        if (i < board.size() - 1) dfs(board, i + 1, j, p, res);
        if (j < board[0].size() - 1) dfs(board, i, j + 1, p, res);
        visit[i][j]=0;
    }
//void main part
vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        vector<string> res;
        trie* root = buildTrie(words);
        int m= board.size(), n = board[0].size();
         memset(visit,0,sizeof(visit));
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                dfs(board,  i, j, root, res);
            }
        }
        return res;
    }

////////////////////////////////////////////////////////////////////////////
Sparse Table (gcd query, min query also the same)

int st[400009][23];
int lg[400009];

void build(int N, vi &a){
    // build of log table
    for (int i=2;i<=400005;i++){
        lg[i]=lg[i/2]+1;
    }
    for (int i = 0; i < N; i++)
    st[i][0] = a[i];
//same build for min also
for (int j = 1; j <= 22; j++)
    for (int i = 0; i + (1 << j) <= N; i++)
        st[i][j] = __gcd(st[i][j-1], st[i + (1 << (j - 1))][j - 1]);
     
}
int query(int l,int r, vi &a){
    if (l==r) return a[l];
    int j=lg[r-l+1];
    return __gcd(st[l][j],st[r-(1<<j)+1][j]);
}
///////////////////////////////////////////////////////////////////////////
Binary exponentiation 
//inverse y=y^mod-2
long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}
///////////////////////////////////////////////////////////////////////
Custom hashmap faster then unordered_map of stl, this gave AC while other
TLE
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
unordered_map<int,int, custom_hash> came;
/////////////////////////////////////////////////////////////////////////
//Articulation point 
//cut vertex

vi adj[10005];
int vis[10005];
int low[10005],tin[10005];
set<int> AP;
int timer=0;

void dfs(int v,int p){
    vis[v]=1;
    low[v]=tin[v]=timer++;
    int child=0;
    for (auto it:adj[v]){
        if (it==p) continue;
        if (vis[it]==0){
            dfs(it,v);
            low[v]=min(low[v],low[it]);
            if (low[it]>=tin[v] && p!=-1){
                AP.insert(v);
                //it is unreachable to anyone before v
            }
            child++;
        }
        else {
            //on the other side (parent side)
            //obvisouly i wont give the lowest time of parent side node to v 
            //becuase lowest time of parent side can be lower than insertion time
            //with which we will compare
            low[v]=min(low[v],tin[it]);
        }
    }
    //cout<<v<<" "<<low[v]<<" "<<tin[v]<<endl;
    if (p==-1 && child>1) AP.insert(v);
    return;

}
////////////////////////////////////////////////////////////////////////
//segment tree recursive  

struct node{
    int val;
    node(){
        val=0;
    }
    node(int x){
        val=x;
    }
};

int n;
int MAXN=8;
node t[(int)32];

 
node combine(node a,node b)
{
    node res;
    res.val=a.val+b.val;
    return res;
}

void build(vi &a, int v, int tl, int tr) {
    if (tl == tr) {
        t[v].val = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm); //left child calculates first half
        build(a, v*2+1, tm+1, tr);
        t[v] = combine(t[v*2], t[v*2+1]); //conbine is function
    }
}
//intailly v=1 (root), tl=0,tr=n-1 (boundaries of the current node);
node query(int v, int tl, int tr, int l, int r) {
    if (l > r)  //out of range
        return 0;
    if (l == tl && r == tr) {
        //if node is fully covered in this
        return t[v];
    }
    int tm = (tl + tr) / 2;
    return combine(query(v*2, tl, tm, l, min(r, tm)), //left child
                query(v*2+1, tm+1, tr, max(l, tm+1), r)); //right child)
}

void update(int v, int tl, int tr, int pos, int new_val) {
    if (tl == tr) {
        t[v] = new_val;
    } else {
        int tm = (tl + tr) / 2;
        if (pos <= tm)
            update(v*2, tl, tm, pos, new_val);
        else
            update(v*2+1, tm+1, tr, pos, new_val);
        t[v] = combine(t[v*2], t[v*2+1]);
    }
}
