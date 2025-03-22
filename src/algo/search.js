// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright © 2025, Kenneth Leung. All rights reserved.

;(function(gscope,UNDEF){

  "use strict";

  /**Create the module.
   */
  function _module(Mcfud,Sort){

    const Basic= Mcfud ? Mcfud["Basic"] : gscope["io/czlab/mcfud/algo/basic"]();
    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();
    const _M = Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();

    if(!Sort) Sort= gscope["io/czlab/mcfud/algo/sort"]();

    const {Bag,Stack,Queue,StdCompare:CMP,prnIter}= Basic;
    const {MinPQ}= Sort;
    const int=Math.floor;
    const {is,u:_}= Core;

    /**
     * @module mcfud/algo_search
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides a client for reading in a sequence of words and printing a word
     * (exceeding a given length) that occurs most frequently.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class FrequencyCounter{
      /**Compute frequency count.
       * @param {array} input the list of words
       * @param {number} keySize the minimum word length
       * @return {array} [max, maxCount, [distinct,words]]
       */
      static count(input,keySize){
        let m=new Map(),
            words=0, max="", distinct=0;
        for(let s,i=0;i<input.length;++i){
          s=input[i];
          if(s.length<keySize)continue;
          ++words;
          if(m.has(s)){
            m.set(s, m.get(s)+1)
          }else{
            m.set(s, 1);
            ++distinct;
          }
        }
        // find a key with the highest frequency count
        m.set(max, 0);
        Array.from(m.keys()).forEach(k=>{
          if(m.get(k) > m.get(max)) max = k;
        });
        return [max, m.get(max),[distinct,words]];
      }
      static test(){
        let s= `it was the best of times it was the worst of times
        it was the age of wisdom it was the age of foolishness
        it was the epoch of belief it was the epoch of incredulity
        it was the season of light it was the season of darkness
        it was the spring of hope it was the winter of despair`.split(" ");
        let [m,v,extra]= FrequencyCounter.count(s,1);
        console.log("" + m + " " + v);
        console.log("distinct = " + extra[0]);
        console.log("words= " + extra[1]);
      }
    }
    //FrequencyCounter.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const SNode=(key,val,next)=> ({key,val,next});

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an (unordered) symbol table of generic key-value pairs.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class SequentialSearchST{
      constructor(){
      //* @property {object} first the linked list of key-value pairs
      //* @property {number} n number of key-value pairs
        this.first=UNDEF;
        this.n=0;
      }
      /**Returns the number of key-value pairs in this symbol table.
       * @return {number}
       */
      size(){
        return this.n;
      }
      /**Returns true if this symbol table is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.size() == 0;
      }
      /**Returns true if this symbol table contains the specified key.
       * @param  {any} key the key
       * @return {boolean}
       */
      contains(key){
        if(_.nichts(key))
          throw Error(`argument to contains is null`);
        return this.get(key) !== undefined;
      }
      /**Returns the value associated with the given key in this symbol table.
       * @param  {any} key the key
       * @return {any}
       */
      get(key){
        if(_.nichts(key))
          throw Error(`argument to get is null`);
        for(let x=this.first; x; x=x.next){
          if(key==x.key)
            return x.val;
        }
      }
      /**Inserts the specified key-value pair into the symbol table, overwriting the old
       * value with the new value if the symbol table already contains the specified key.
       * Deletes the specified key (and its associated value) from this symbol table
       * if the specified value is {@code null}.
       * @param  {any} key the key
       * @param  {any} val the value
       */
      put(key, val){
        if(_.nichts(key))
          throw Error(`first argument to put is null`);
        if(val === undefined){
          this.delete(key);
        }else{
          let f,x;
          for(x=this.first; x && !f; x=x.next){
            if(key==x.key){
              x.val = val;
              f=true;
            }
          }
          if(!f){//add to head
            this.first = SNode(key, val, this.first);
            this.n +=1;
          }
        }
      }
      /**Removes the specified key and its associated value from this symbol table
       * (if the key is in this symbol table).
       * @param  {any} key the key
       */
      delete(key){
        // delete key in linked list beginning at Node x
        // warning: function call stack too large if table is large
        const _delete=(x,key)=>{
          if(!x) return UNDEF;
          if(key==x.key){
            this.n -= 1;
            return x.next;
          }
          x.next = _delete(x.next, key);
          return x;
        }
        if(_.nichts(key))
          throw Error(`argument to delete is null`);
        this.first = _delete(this.first, key);
      }
      /**Returns all keys in the symbol table as an {@code Iterable}.
       * To iterate over all of the keys in the symbol table named {@code st},
       * use the foreach notation: {@code for (Key key : st.keys())}.
       * @return {Iterator}
       */
      keys(){
        let out=new Queue();
        for(let x=this.first; x; x=x.next) out.enqueue(x.key);
        return out.iter();
      }
      static load(input){
        let obj=new SequentialSearchST();
        input.forEach((s,i)=> obj.put(s,i));
        return obj
      }
      static test(){
        let obj=SequentialSearchST.load("SEARCHEXAMPLE".split(""));
        let fn=(s="",k=0,it=0)=>{
          for(it=obj.keys(); it.hasNext();){
            k=it.next(); s+=`${k}=${obj.get(k)} `; } return s};
        console.log(fn());
        console.log("size= " + obj.size());
        console.log("contains R= " + obj.contains("R"));
        console.log("get R= " + obj.get("R"));
        obj.delete("R");
        obj.isEmpty();
        console.log("contains R= " + obj.contains("R"));
        console.log("get R= " + obj.get("R"));
        console.log("size= " + obj.size());
      }
    }
    //SequentialSearchST.test();

    /**Represents an ordered symbol table of generic key-value pairs.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class BinarySearchST{
      /**Initializes an empty symbol table with the specified initial capacity.
       * @param {function} compareFn
       * @param {number} capacity
       */
      constructor(compareFn,capacity=2){
        //* @property {array} mKeys
        //* @property {array} vals
        //* @property {number} n
        //* @property {function} compare
        this.mKeys= new Array(capacity);
        this.vals= new Array(capacity);
        this.compare=compareFn;
        this.n=0;
        //resize the underlying arrays
        this._resize=(c)=>{
          let tempk = new Array(c),
              tempv = new Array(c);
          for(let i=0; i<this.n; ++i){
            tempk[i] = this.mKeys[i];
            tempv[i] = this.vals[i];
          }
          this.vals = tempv;
          this.mKeys = tempk;
        };
        this._argOk=(p)=> _.echt(p, "Invalid argument");
        this._check=()=>{
          const isSorted=()=>{
            // are the items in the array in ascending order?
            for(let i=1; i<this.size(); ++i)
              if(this.compare(this.mKeys[i],this.mKeys[i-1])<0) return false;
            return true;
          };
          const rankCheck=()=>{
            // check that rank(select(i)) = i
            for(let i=0; i<this.size(); ++i)
              if(i != this.rank(this.select(i))) return false;
            for(let i=0; i<this.size(); ++i)
              if(this.compare(this.mKeys[i],this.select(this.rank(this.mKeys[i]))) != 0) return false;
            return true;
          };
          return isSorted() && rankCheck();
        };
      }
      /**Returns true if this symbol table is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.size() == 0;
      }
      /**Does this symbol table contain the given key?
       * @param  key the key
       * @return {boolean}
       */
      contains(key){
        return this._argOk(key) && this.get(key) !== undefined
      }
      /**Returns the value associated with the given key in this symbol table.
       * @param  {any} key the key
       * @return {any}
       */
      get(key){
        if(this._argOk(key) && !this.isEmpty()){
          let i = this.rank(key);
          if(i<this.n &&
             this.compare(this.mKeys[i],key) == 0) return this.vals[i];
        }
      }
      /**Returns the number of keys in this symbol table strictly less than {@code key}.
       * @param  {any} key the key
       * @return {number}
       */
      rank(key){
        let mid,cmp,
            lo=0, hi=this.n-1;
        this._argOk(key);
        while(lo <= hi){
          mid = lo+ _M.ndiv(hi-lo,2);
          cmp = this.compare(key,this.mKeys[mid]);
          if(cmp < 0) hi = mid-1;
          else if(cmp > 0) lo = mid+1;
          else return mid;
        }
        return lo;
      }
      /**Inserts the specified key-value pair into the symbol table, overwriting the old
       * value with the new value if the symbol table already contains the specified key.
       * Deletes the specified key (and its associated value) from this symbol table
       * if the specified value is {@code null}.
       * @param  {any} key the key
       * @param  {any} val the value
       */
      put(key, val){
        if(this._argOk(key) && val===undefined){
          this.delete(key);
        }else{
          let i=this.rank(key);
          // key is already in table
          if(i<this.n && this.compare(this.mKeys[i],key) == 0){
            this.vals[i] = val;
          }else{
            // insert new key-value pair
            if(this.n == this.mKeys.length)
              this._resize(2*this.mKeys.length);
            for(let j=this.n; j>i; --j){
              this.mKeys[j] = this.mKeys[j-1];
              this.vals[j] = this.vals[j-1];
            }
            this.n+=1;
            this.mKeys[i] = key;
            this.vals[i] = val;
            //this._check();
          }
        }
      }
      /**Removes the specified key and associated value from this symbol table
       * (if the key is in the symbol table).
       * @param  {any} key the key
       */
      delete(key){
        if(this._argOk(key) && this.isEmpty()){
        }else{
          // compute rank
          let i=this.rank(key);
          // key not in table
          if(i==this.n || this.compare(this.mKeys[i],key) != 0){
          }else{
            for(let j=i; j<this.n-1; ++j){
              this.mKeys[j] = this.mKeys[j+1];
              this.vals[j] = this.vals[j+1];
            }
            this.n-=1;
            this.mKeys[this.n] = UNDEF;  // to avoid loitering
            this.vals[this.n] = UNDEF;
            // resize if 1/4 full
            if(this.n>0 &&
               this.n == _M.ndiv(this.mKeys.length,4))
              this._resize(_M.ndiv(this.mKeys.length,2));
            this._check();
          }
        }
      }
      /**Removes the smallest key and associated value from this symbol table.
       */
      deleteMin(){
        if(this.isEmpty())
          throw Error(`Symbol table underflow error`);
        this.delete(this.min());
      }
      /**Removes the largest key and associated value from this symbol table.
       */
      deleteMax(){
        if(this.isEmpty())
          throw Error(`Symbol table underflow error`);
        this.delete(this.max());
      }
      /**Returns the smallest key in this symbol table.
       * @return {any}
       */
      min(){
        if(this.isEmpty())
          throw Error(`called min with empty symbol table`);
        return this.mKeys[0];
      }
      /**Returns the largest key in this symbol table.
       * @return {any}
       */
      max(){
        if(this.isEmpty())
          throw Error(`called max with empty symbol table`);
        return this.mKeys[this.n-1];
      }
      /**Return the kth smallest key in this symbol table.
       * @param  {number} k the order statistic
       * @return {any}
       */
      select(k){
        if(k<0 || k>=this.size())
          throw Error(`called select with invalid argument: ${k}`);
        return this.mKeys[k];
      }
      /**Returns the largest key in this symbol table less than or equal to {@code key}.
       * @param  {any} key the key
       * @return {any}
       */
      floor(key){
        let i = this._argOk(key) && this.rank(key);
        if(i<this.n &&
           this.compare(key,this.mKeys[i]) == 0)
          return this.mKeys[i];
        if(i==0)
          throw Error(`argument to floor is too small`);
        return this.mKeys[i-1];
      }
      /**Returns the smallest key in this symbol table greater than or equal to {@code key}.
       * @param  {any} key the key
       * @return {any}
       */
      ceiling(key){
        let i=this._argOk(key) && this.rank(key);
        if(i==this.n)
          throw Error(`argument to ceiling is too large`);
        return this.mKeys[i];
      }
      /**Returns the number of keys in this symbol table in the specified range.
       * @param {number} lo minimum endpoint
       * @param {number} hi maximum endpoint
       * @return {number} the number of keys in this symbol table between {@code lo}
       *                  (inclusive) and {@code hi} (inclusive)
       */
      size(lo, hi){
        if(arguments.length==0){
          return this.n;
        }
        this._argOk(lo) && this._argOk(hi);
        return this.compare(lo,hi)>0 ?0
                                     :(this.contains(hi)?(this.rank(hi)-this.rank(lo)+1)
                                                        :(this.rank(hi)-this.rank(lo)));
      }
      /**Returns all keys in this symbol table in the given range,
       * as an {@code Iterable}.
       * @param lo minimum endpoint
       * @param hi maximum endpoint
       * @return {Iterator} all keys in this symbol table between {@code lo}
       *                    (inclusive) and {@code hi} (inclusive)
       */
      keys(lo, hi){
        if(arguments.length==0){
          lo=this.min();
          hi=this.max();
        }
        this._argOk(lo) && this._argOk(hi);
        let out=new Queue();
        if(this.compare(lo,hi) > 0){}else{
          for(let i=this.rank(lo); i<this.rank(hi); ++i)
            out.enqueue(this.mKeys[i]);
          if(this.contains(hi))
            out.enqueue(this.mKeys[this.rank(hi)]);
        }
        return out.iter();
      }
      static load(input,compareFn){
        let obj= new BinarySearchST(compareFn);
        input.forEach((s,i)=> obj.put(s,i));
        return obj;
      }
      static test(){
        let b= BinarySearchST.load("SEARCHEXAMPLE".split(""),CMP);
        let fn=(s)=>{s="";
          for(let k,it=b.keys();it.hasNext();){
            k=it.next(); s+=`${k}=${b.get(k)} ` } return s}
        console.log(fn());
        b.deleteMin();
        console.log(fn());
        b.deleteMax();
        b.isEmpty();
        console.log(fn());
        console.log("floor of Q= " + b.floor("Q"));
        console.log("ceil of Q= " + b.ceiling("Q"));
        console.log("size= " + b.size());
        console.log("size= " + b.size("E","P"));
        console.log("keys E->P = " + prnIter(b.keys("E","P")));
      }
    }
    //BinarySearchST.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an ordered symbol table of generic key-value pairs.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class BST{
      constructor(compareFn){
        //* @property {object} root
        //* @property {function} compare
        this.compare=compareFn;
        this.root=UNDEF;
        this._argOk=(x)=> _.assert(x, "Invalid argument");
        this._check=()=>{
          if(!this.isBST(this.root,null,null)) console.log("Not in symmetric order");
          if(!this.isSizeConsistent(this.root)) console.log("Subtree counts not consistent");
          if(!this.isRankConsistent()) console.log("Ranks not consistent");
          return this.isBST(this.root,null,null) && this.isSizeConsistent(this.root) && this.isRankConsistent();
        };
        // is the tree rooted at x a BST with all keys strictly between min and max
        // (if min or max is null, treat as empty constraint)
        // Credit: Bob Dondero's elegant solution
        this.isBST=(x, min, max)=>{
          if(_.nichts(x)) return true;
          if(_.echt(min) && this.compare(x.key,min) <= 0) return false;
          if(_.echt(max) && this.compare(x.key,max) >= 0) return false;
          return this.isBST(x.left, min, x.key) && this.isBST(x.right, x.key, max);
        };
        this.isSizeConsistent=(x)=>{
          if(_.nichts(x)) return true;
          if(x.size != (this._sizeNode(x.left) + this._sizeNode(x.right) + 1)) return false;
          return this.isSizeConsistent(x.left) && this.isSizeConsistent(x.right);
        };
        // check that ranks are consistent
        this.isRankConsistent=()=>{
          for(let i=0; i<this.size(); ++i)
            if(i != this.rank(this.select(i))) return false;
          for(let k, it=this.keys(); it.hasNext();){
            k=it.next();
            if(this.compare(k,this.select(this.rank(k))) != 0) return false;
          }
          return true;
        };
      }
      Node(key, val, size){
        return{ key,val,size};// left:null, right:null
      }
      /**Returns true if this symbol table is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.size() == 0;
      }
      /**Does this symbol table contain the given key?
       * @param  {any} key the key
       * @return {boolean}
       */
      contains(key){
        return this._argOk(key) && this.get(key) !== undefined;
      }
      /**Returns the value associated with the given key.
       * @param  {any} key the key
       * @return {any}
       */
      get(key){
        return this._getNode(this.root, key);
      }
      _getNode(x, key){
        if(this._argOk(key) && _.nichts(x)){}else{
          let cmp = this.compare(key,x.key);
          return cmp < 0? this._getNode(x.left, key) :(cmp > 0? this._getNode(x.right, key) : x.val)
        }
      }
      /**Inserts the specified key-value pair into the symbol table, overwriting the old
       * value with the new value if the symbol table already contains the specified key.
       * Deletes the specified key (and its associated value) from this symbol table
       * if the specified value is {@code null}.
       * @param  {any} key the key
       * @param  {any} val the value
       */
      put(key, val){
        if(this._argOk(key) && _.nichts(val)){
          this.delete(key);
        }else{
          this.root = this._putNode(this.root, key, val);
          this._check();
        }
      }
      _putNode(x, key, val){
        if(_.nichts(x)){ x=this.Node(key, val, 1) }else{
          let cmp = this.compare(key,x.key);
          if(cmp < 0) x.left = this._putNode(x.left,  key, val);
          else if(cmp > 0) x.right = this._putNode(x.right, key, val);
          else x.val = val;
          x.size = 1 + this._sizeNode(x.left) + this._sizeNode(x.right);
        }
        return x;
      }
      /**Removes the smallest key and associated value from the symbol table.
       */
      deleteMin(){
        if(this.isEmpty()) throw Error("Symbol table underflow");
        this.root = this._deleteMinNode(this.root);
        this._check();
      }
      _deleteMinNode(x){
        if(_.nichts(x.left)){ x= x.right }else{
          x.left = this._deleteMinNode(x.left);
          x.size = this._sizeNode(x.left) + this._sizeNode(x.right) + 1;
        }
        return x;
      }
      /**Removes the largest key and associated value from the symbol table.
       */
      deleteMax(){
        if(this.isEmpty()) throw Error("Symbol table underflow");
        this.root = this._deleteMaxNode(this.root);
        this._check();
      }
      _deleteMaxNode(x){
        if(_.nichts(x.right)){ x= x.left }else{
          x.right = this._deleteMaxNode(x.right);
          x.size = this._sizeNode(x.left) + this._sizeNode(x.right) + 1;
        }
        return x;
      }
      /**Removes the specified key and its associated value from this symbol table
       * (if the key is in this symbol table).
       * @param  {any} key the key
       */
      delete(key){
        this.root= this._argOk(key) && this._deleteNode(this.root, key);
        this._check();
      }
      _deleteNode(x, key){
        if(_.echt(x)){
          let cmp = this.compare(key,x.key);
          if(cmp < 0) x.left = this._deleteNode(x.left,  key);
          else if(cmp > 0) x.right = this._deleteNode(x.right, key);
          else{
            if(_.nichts(x.right)) return x.left;
            if(_.nichts(x.left)) return x.right;
            let t = x;
            x= this._minNode(t.right);
            x.right = this._deleteMinNode(t.right);
            x.left = t.left;
          }
          x.size = this._sizeNode(x.left) + this._sizeNode(x.right) + 1;
        }
        return x;
      }
      /**Returns the smallest key in the symbol table.
       * @return {any}
       */
      min(){
        if(this.isEmpty())
          throw Error(`calls min with empty symbol table`);
        return this._minNode(this.root).key;
      }
      _minNode(x){
        return _.nichts(x.left)? x: this._minNode(x.left);
      }
      /**Returns the largest key in the symbol table.
       * @return {any}
       */
      max(){
        if(this.isEmpty())
          throw Error(`calls max with empty symbol table`);
        return this._maxNode(this.root).key;
      }
      _maxNode(x){
        return _.nichts(x.right)? x: this._maxNode(x.right);
      }
      /**Returns the largest key in the symbol table less than or equal to {@code key}.
       * @param  {any} key the key
       * @return {any}
       */
      floor(key){
        if(this._argOk(key) && this.isEmpty())
          throw Error(`calls floor with empty symbol table`);
        let x= this._floorNode(this.root, key);
        if(_.nichts(x))
          throw Error(`argument to floor is too small`);
        return x.key;
      }
      _floorNode(x, key){
        if(_.nichts(x)){ return null }
        let cmp = this.compare(key,x.key);
        if(cmp == 0) return x;
        if(cmp < 0) return this._floorNode(x.left, key);
        let t = this._floorNode(x.right, key);
        return _.nichts(t)?x: t;
      }
      /**Returns the smallest key in the symbol table greater than or equal to {@code key}.
       * @param  {any} key the key
       * @return {any}
       */
      ceiling(key){
        if(this._argOk(key) && this.isEmpty())
          throw Error(`calls ceiling with empty symbol table`);
        let x = this._ceilingNode(this.root, key);
        if(_.nichts(x))
          throw Error(`argument to floor is too large`);
        return x.key;
      }
      _ceilingNode(x, key){
        if(_.nichts(x)){return UNDEF}
        let cmp = this.compare(key,x.key);
        if(cmp == 0) return x;
        if(cmp < 0){
          let t = this._ceilingNode(x.left, key);
          return t? t: x;
        }
        return this._ceilingNode(x.right, key);
      }
      /**Return the key in the symbol table of a given {@code rank}.
       * This key has the property that there are {@code rank} keys in
       * the symbol table that are smaller. In other words, this key is the
       * ({@code rank}+1)st smallest key in the symbol table.
       * @param  {number} rank the order statistic
       * @return {any}
       */
      select(rank){
        if(rank < 0 || rank >= this.size())
          throw Error(`argument to select is invalid: ${rank}`);
        return this._selectNode(this.root, rank);
      }
      // Return key in BST rooted at x of given rank.
      // Precondition: rank is in legal range.
      _selectNode(x, rank){
        if(_.nichts(x)){return UNDEF}
        let leftSize = this._sizeNode(x.left);
        if(leftSize > rank) return this._selectNode(x.left,  rank);
        if(leftSize < rank) return this._selectNode(x.right, rank - leftSize - 1);
        return x.key;
      }
      /**Return the number of keys in the symbol table strictly less than {@code key}.
       * @param  {any} key the key
       * @return {number}
       */
      rank(key){
        return this._argOk(key) && this._rankNode(key, this.root);
      }
      // Number of keys in the subtree less than key.
      _rankNode(key, x){
        if(_.nichts(x)){return 0}
        let cmp = this.compare(key,x.key);
        return cmp < 0? this._rankNode(key, x.left)
                      : (cmp > 0? (1 + this._sizeNode(x.left) + this._rankNode(key, x.right)) :this._sizeNode(x.left));
      }
      /**Returns all keys in the symbol table in the given range,
       * as an {@code Iterable}.
       * @param  {any} lo minimum endpoint
       * @param  {any} hi maximum endpoint
       * @return {Iterator} all keys in the symbol table between {@code lo}
       *         (inclusive) and {@code hi} (inclusive)
       */
      keys(lo, hi){
        let Q=new Queue();
        if(arguments.length==0){
          if(!this.isEmpty()){
            lo=this.min();
            hi=this.max();
          }
        }
        if(!this.isEmpty() && this._argOk(lo) && this._argOk(hi)){
          this._keysNode(this.root, Q, lo, hi);
        }
        return Q.iter();
      }
      _keysNode(x, queue, lo, hi){
        if(_.nichts(x)){}else{
          let cmplo = this.compare(lo,x.key);
          let cmphi = this.compare(hi,x.key);
          if(cmplo < 0) this._keysNode(x.left, queue, lo, hi);
          if(cmplo <= 0 && cmphi >= 0) queue.enqueue(x.key);
          if(cmphi > 0) this._keysNode(x.right, queue, lo, hi);
        }
        return queue;
      }
      // return number of key-value pairs in BST rooted at x
      _sizeNode(x){
        return _.nichts(x)?0: x.size;
      }
      /**Returns the number of keys in the symbol table in the given range.
       * @param  {any} lo minimum endpoint
       * @param  {any} hi maximum endpoint
       * @return {number} number of keys in the symbol table between {@code lo}
       *         (inclusive) and {@code hi} (inclusive)
       */
      size(lo, hi){
        return arguments.length==0 ? this._sizeNode(this.root)
          : ( (this._argOk(lo) &&
                this._argOk(hi) &&
                this.compare(lo,hi)>0)? 0
                                      : (this.contains(hi)? (this.rank(hi) - this.rank(lo) + 1): (this.rank(hi) - this.rank(lo))));
      }
      /**Returns the height of the BST (for debugging).
       * @return {number} the height of the BST (a 1-node tree has height 0)
       */
      height(){
        return this._heightNode(this.root)
      }
      _heightNode(x){
        return _.nichts(x)? -1 : (1 + Math.max(this._heightNode(x.left), this._heightNode(x.right)))
      }
      /**Returns the keys in the BST in level order (for debugging).
       * @return {Iterator} the keys in the BST in level order traversal
       */
      levelOrder(){
        let x,queue = [],
            keys = new Queue();
        queue.push(this.root);
        while(queue.length>0){
          x = queue.pop();
          if(_.echt(x)){
            keys.enqueue(x.key);
            queue.push(x.left, x.right);
          }
        }
        return keys.iter();
      }
      static load(input,compareFn){
        let b=new BST(compareFn);
        input.forEach((s,i)=> b.put(s,i));
        return b;
      }
      static test(){
        let m,obj= BST.load("SEARCHEXAMPLE".split(""),CMP);
        m="";
        for(let s,it=obj.levelOrder(); it.hasNext();){
          s=it.next(); m+= `${s}=${obj.get(s)} `
        }
        console.log("level-order:\n"+m);
        m="";
        for(let s,it=obj.keys(); it.hasNext();){
          s=it.next(); m+= `${s}=${obj.get(s)} `
        }
        obj.isEmpty();
        console.log("keys=\n"+m);
        console.log("size="+obj.size());
        console.log("size E->Q = ", obj.size("E","Q"));
        m="";
        for(let s,it=obj.keys("E","Q"); it.hasNext();){
          s=it.next(); m+= `${s}=${obj.get(s)} `
        }
        console.log("keys[E->Q]= "+m);
        console.log("min= "+obj.min());
        console.log("max= "+obj.max());
        console.log("rank P= " +obj.rank("P"));
        console.log("contains X= "+obj.contains("X"));
        console.log("contains Z= "+obj.contains("Z"));
        obj.delete("X");
        console.log("get C=" + obj.get("C"));
        console.log("max= "+obj.max());
        obj.deleteMin();
        obj.deleteMax();
        console.log("height= " +obj.height());
        console.log("min= "+obj.min());
        console.log("max= "+obj.max());
        console.log("rank E= "+obj.rank("E"));
        console.log("floor G= " +obj.floor("G"));
        console.log("ceiling G= " +obj.ceiling("G"));
      }
    }
    //BST.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an ordered symbol table of generic key-value pairs.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class RedBlackBST{
      static BLACK = false;
      static RED= true;
      constructor(compareFn){
        //* @property {object} root
        //* @property {function} compare
        this.compare=compareFn;
        this.root=UNDEF;
        this._argOk=(x)=>_.assert(x, "Invalid argument");
        this._check=()=>{
          // is the tree rooted at x a BST with all keys strictly between min and max
          // (if min or max is null, treat as empty constraint)
          // Credit: Bob Dondero's elegant solution
          let isBST3=(x, min, max)=>{
            if(_.nichts(x)) return true;
            if(min && this.compare(x.key,min) <= 0) return false;
            if(max && this.compare(x.key,max) >= 0) return false;
            return isBST3(x.left, min, x.key) && isBST3(x.right, x.key, max);
          };
          let isSizeConsistent=(x)=>{
            if(_.nichts(x)) return true;
            if(x.size != this._sizeNode(x.left) + this._sizeNode(x.right) + 1) return false;
            return isSizeConsistent(x.left) && isSizeConsistent(x.right);
          }
          // check that ranks are consistent
          let isRankConsistent=()=>{
            for(let i=0; i<this.size(); ++i)
              if(i != this._rankNode(this.select(i))) return false;
            for(let k,it=this.keys(); it.hasNext();){
              k=it.next();
              if(this.compare(k,this.select(this._rankNode(k))) != 0) return false;
            }
            return true;
          };
          // Does the tree have no red right links, and at most one (left)
          // red links in a row on any path?
          let is23=(x)=>{
            if(_.nichts(x)) return true;
            if(this._isRed(x.right)) return false;
            if (x !== this.root && this._isRed(x) && this._isRed(x.left)) return false;
            return is23(x.left) && is23(x.right);
          }
          // do all paths from root to leaf have same number of black edges?
          let isBalanced=()=>{
            let black = 0,// number of black links on path from root to min
                x = this.root;
            while(x){
              if(!this._isRed(x)) ++black;
              x=x.left;
            }
            return isBalanced2(this.root, black);
          };
          // does every path from the root to a leaf have the given number of black links?
          let isBalanced2=(x, black)=>{
            if(_.nichts(x)) return black == 0;
            if(!this._isRed(x)) --black;
            return isBalanced2(x.left, black) && isBalanced2(x.right, black);
          };
          return isBST3(this.root,null,null) && isSizeConsistent(this.root) && isRankConsistent() && is23(this.root) && isBalanced();
        };
      }
      Node(key, val, color, size){
        //color is parent color
        return {key,val,color,size};//left:null,right:null
      }
      // is node x red; false if x is null ?
      _isRed(x){
        return _.nichts(x)?false:x.color=== RedBlackBST.RED
      }
      //number of node in subtree rooted at x; 0 if x is null
      _sizeNode(x){
        return _.nichts(x)?0:x.size
      }
      /**Is this symbol table empty?
       * @return {boolean}
       */
      isEmpty(){
        return _.nichts(this.root)
      }
      /**Returns the value associated with the given key.
       * @param {any} key the key
       * @return {any}
       */
      get(key){
        return this._argOk(key) && this._getNode(this.root, key);
      }
      // value associated with the given key in subtree rooted at x; null if no such key
      _getNode(x, key){
        while(x){
          let cmp = this.compare(key,x.key);
          if(cmp < 0) x = x.left;
          else if(cmp > 0) x = x.right;
          else return x.val;
        }
      }
      /**Does this symbol table contain the given key?
       * @param {any} key the key
       * @return {boolean}
       */
      contains(key){
        return this.get(key) !== undefined
      }
      /***************************************************************************
       *  Red-black tree insertion.
       ***************************************************************************/
      /**Inserts the specified key-value pair into the symbol table, overwriting the old
       * value with the new value if the symbol table already contains the specified key.
       * Deletes the specified key (and its associated value) from this symbol table
       * if the specified value is {@code null}.
       * @param {any} key the key
       * @param {any} val the value
       */
      put(key, val){
        if(this._argOk(key) && _.nichts(val)){
          this.delete(key);
        }else{
          this.root = this._putNode(this.root, key, val);
          this.root.color = RedBlackBST.BLACK;
        }
      }
      // insert the key-value pair in the subtree rooted at h
      _putNode(h, key, val){
        if(_.nichts(h)) return this.Node(key, val, RedBlackBST.RED, 1);
        let cmp = this.compare(key,h.key);
        if(cmp < 0) h.left  = this._putNode(h.left, key, val);
        else if(cmp > 0) h.right = this._putNode(h.right, key, val);
        else h.val = val;
        // fix-up any right-leaning links
        if(this._isRed(h.right) && !this._isRed(h.left))  h = this._rotateLeft(h);
        if(this._isRed(h.left)  &&  this._isRed(h.left.left)) h = this._rotateRight(h);
        if(this._isRed(h.left)  &&  this._isRed(h.right)) this._flipColors(h);
        h.size = this._sizeNode(h.left) + this._sizeNode(h.right) + 1;
        return h;
      }
      /***************************************************************************
       *  Red-black tree deletion.
       ***************************************************************************/
      /**Removes the smallest key and associated value from the symbol table.
       */
      deleteMin(){
        if(this.isEmpty())
          throw Error("BST underflow");
        // if both children of root are black, set root to red
        if(!this._isRed(this.root.left) &&
           !this._isRed(this.root.right))
          this.root.color = RedBlackBST.RED;
        this.root = this._deleteMinNode(this.root);
        if(!this.isEmpty()) this.root.color = RedBlackBST.BLACK;
      }
      // delete the key-value pair with the minimum key rooted at h
      _deleteMinNode(h){
        if(_.nichts(h.left)) return null;
        if(!this._isRed(h.left) &&
           !this._isRed(h.left.left))
          h = this._moveRedLeft(h);
        h.left = this._deleteMinNode(h.left);
        return this._balance(h);
      }
      /**Removes the largest key and associated value from the symbol table.
       */
      deleteMax(){
        if(this.isEmpty())
          throw Error("BST underflow");
        // if both children of root are black, set root to red
        if(!this._isRed(this.root.left) &&
           !this._isRed(this.root.right))
          this.root.color = RedBlackBST.RED;
        this.root = this._deleteMaxNode(this.root);
        if(!this.isEmpty()) this.root.color = RedBlackBST.BLACK;
      }
      // delete the key-value pair with the maximum key rooted at h
      _deleteMaxNode(h){
        if(this._isRed(h.left)) h = this._rotateRight(h);
        if(_.nichts(h.right)) return null;
        if(!this._isRed(h.right) &&
           !this._isRed(h.right.left))
          h = this._moveRedRight(h);
        h.right = this._deleteMaxNode(h.right);
        return this._balance(h);
      }
      /**Removes the specified key and its associated value from this symbol table
       * (if the key is in this symbol table).
       * @param  {any} key the key
       */
      delete(key){
        if(this._argOk(key) && !this.contains(key)){}else{
          //if both children of root are black, set root to red
          if(!this._isRed(this.root.left) &&
             !this._isRed(this.root.right)) this.root.color = RedBlackBST.RED;
          this.root = this._deleteNode(this.root, key);
          if(!this.isEmpty()) this.root.color = RedBlackBST.BLACK;
        }
        //this._check();
      }
      // delete the key-value pair with the given key rooted at h
      _deleteNode(h, key){
        if(this.compare(key,h.key) < 0){
          if(!this._isRed(h.left) &&
             !this._isRed(h.left.left))
            h = this._moveRedLeft(h);
          h.left = this._deleteNode(h.left, key);
        }else{
          if(this._isRed(h.left))
            h = this._rotateRight(h);
          if(this.compare(key,h.key) == 0 &&
             _.nichts(h.right)) return null;
          if(!this._isRed(h.right) &&
             !this._isRed(h.right.left))
            h = this._moveRedRight(h);
          if(this.compare(key,h.key) == 0){
            let x = this._minNode(h.right);
            h.key = x.key;
            h.val = x.val;
            h.right = this._deleteMinNode(h.right);
          }else{
            h.right = this._deleteNode(h.right, key);
          }
        }
        return this._balance(h);
      }
      /***************************************************************************
       *  Red-black tree helper functions.
       ***************************************************************************/
      // make a left-leaning link lean to the right
      _rotateRight(h){//console.log("_RR");
        if(_.nichts(h) || !this._isRed(h.left))
          throw Error("bad input to rotateRight");
        let x = h.left;
        h.left = x.right;
        x.right = h;
        x.color = x.right.color;
        x.right.color = RedBlackBST.RED;
        x.size = h.size;
        h.size = this._sizeNode(h.left) + this._sizeNode(h.right) + 1;
        return x;
      }
      // make a right-leaning link lean to the left
      _rotateLeft(h){//console.log("RL");
        if(_.nichts(h) || !this._isRed(h.right))
          throw Error("bad input to rotateLeft");
        let x = h.right;
        h.right = x.left;
        x.left = h;
        x.color = x.left.color;
        x.left.color = RedBlackBST.RED;
        x.size = h.size;
        h.size = this._sizeNode(h.left) + this._sizeNode(h.right) + 1;
        return x;
      }
      // flip the colors of a node and its two children
      _flipColors(h){//console.log("FF");
        // h must have opposite color of its two children
        // assert (h != null) && (h.left != null) && (h.right != null);
        // assert (!isRed(h) &&  isRed(h.left) &&  isRed(h.right))
        //    || (isRed(h)  && !isRed(h.left) && !isRed(h.right));
        h.color = !h.color;
        h.left.color = !h.left.color;
        h.right.color = !h.right.color;
      }
      // Assuming that h is red and both h.left and h.left.left
      // are black, make h.left or one of its children red.
      _moveRedLeft(h){//console.log("MoveRL");
        // assert (h != null);
        // assert isRed(h) && !isRed(h.left) && !isRed(h.left.left);
        this._flipColors(h);
        if(this._isRed(h.right.left)){
          h.right = this._rotateRight(h.right);
          h = this._rotateLeft(h);
          this._flipColors(h);
        }
        return h;
      }
      // Assuming that h is red and both h.right and h.right.left
      // are black, make h.right or one of its children red.
      _moveRedRight(h){//console.log("MoveRR");
        // assert (h != null);
        // assert isRed(h) && !isRed(h.right) && !isRed(h.right.left);
        this._flipColors(h);
        if(this._isRed(h.left.left)){
          h = this._rotateRight(h);
          this._flipColors(h);
        }
        return h;
      }
      // restore red-black tree invariant
      _balance(h){//console.log("BAL");
        // assert (h != null);
        if(this._isRed(h.right) && !this._isRed(h.left))    h = this._rotateLeft(h);
        if(this._isRed(h.left) && this._isRed(h.left.left)) h = this._rotateRight(h);
        if(this._isRed(h.left) && this._isRed(h.right))     this._flipColors(h);
        h.size = this._sizeNode(h.left) + this._sizeNode(h.right) + 1;
        return h;
      }
      /**Returns the height of the BST (for debugging).
       * @return {number}
       */
      height(){
        return this._height(this.root);
      }
      _height(x){
        return _.nichts(x)? -1: (1 + Math.max(this._height(x.left), this._height(x.right)));
      }
      /**Returns the smallest key in the symbol table.
       * @return {any}
       */
      min(){
        if(this.isEmpty())
          throw Error(`calls min with empty symbol table`);
        return this._minNode(this.root).key;
      }
      // the smallest key in subtree rooted at x; null if no such key
      _minNode(x){
        return _.nichts(x.left)? x: this._minNode(x.left);
      }
      /**Returns the largest key in the symbol table.
       * @return {any}
       */
      max(){
        if(this.isEmpty())
          throw Error(`calls max with empty symbol table`);
        return this._maxNode(this.root).key;
      }
      // the largest key in the subtree rooted at x; null if no such key
      _maxNode(x){
        return _.nichts(x.right)? x : this._maxNode(x.right);
      }
      /**Returns the largest key in the symbol table less than or equal to {@code key}.
       * @param {any} key the key
       * @return {any}
       */
      floor(key){
        if(this._argOk(key) && this.isEmpty())
          throw Error(`calls floor with empty symbol table`);
        let x = this._floorNode(this.root, key);
        if(_.nichts(x))
          throw Error(`argument to floor is too small`);
        return x.key;
      }
      // the largest key in the subtree rooted at x less than or equal to the given key
      _floorNode(x, key){
        if(_.nichts(x)) return UNDEF;
        let cmp = this.compare(key,x.key);
        if(cmp == 0) return x;
        if(cmp < 0)  return this._floorNode(x.left, key);
        let t = this._floorNode(x.right, key);
        return t? t: x;
      }
      /**Returns the smallest key in the symbol table greater than or equal to {@code key}.
       * @param {any} key the key
       * @return {any}
       */
      ceiling(key){
        if(this._argOk(key) && this.isEmpty())
          throw Error(`calls ceiling with empty symbol table`);
        let x = this._ceilingNode(this.root, key);
        if(_.nichts(x))
          throw Error(`argument to ceiling is too small`);
        return x.key;
      }
      // the smallest key in the subtree rooted at x greater than or equal to the given key
      _ceilingNode(x, key){
        if(_.nichts(x)) return UNDEF;
        let cmp = this.compare(key,x.key);
        if(cmp == 0) return x;
        if(cmp > 0)  return this._ceilingNode(x.right, key);
        let t = this._ceilingNode(x.left, key);
        return t? t: x;
      }
      /**Return the key in the symbol table of a given {@code rank}.
       * This key has the property that there are {@code rank} keys in
       * the symbol table that are smaller. In other words, this key is the
       * ({@code rank}+1)st smallest key in the symbol table.
       * @param  {number} rank the order statistic
       * @return {any}
       */
      select(rank){
        if(rank < 0 || rank >= this.size())
          throw Error(`argument to select is invalid: ${rank}`);
        return this._selectNode(this.root, rank);
      }
      // Return key in BST rooted at x of given rank.
      // Precondition: rank is in legal range.
      _selectNode(x, rank){
        if(_.nichts(x)) return UNDEF;
        let leftSize = this._sizeNode(x.left);
        return leftSize > rank? this._selectNode(x.left,  rank)
                              : (leftSize < rank? this._selectNode(x.right, rank - leftSize - 1): x.key);
      }
      /**Return the number of keys in the symbol table strictly less than {@code key}.
       * @param {any} key the key
       * @return {number}
       */
      rank(key){
        return this._argOk(key) && this._rankNode(key, this.root);
      }
      // number of keys less than key in the subtree rooted at x
      _rankNode(key, x){
        if(_.nichts(x)) return 0;
        let cmp = this.compare(key,x.key);
        return cmp < 0? this._rankNode(key, x.left)
                      :(cmp > 0? (1 + this._sizeNode(x.left) + this._rankNode(key, x.right)) :  this._sizeNode(x.left));
      }
      /**Returns all keys in the symbol table in the given range,
       * as an {@code Iterable}.
       * @param  {any} lo minimum endpoint
       * @param  {any} hi maximum endpoint
       * @return {Iterator} all keys in the symbol table between {@code lo}
       *    (inclusive) and {@code hi} (inclusive) as an {@code Iterable}
       */
      keys(lo, hi){
        let Q=new Queue();
        if(arguments.length==0){
          if(!this.isEmpty()){
            lo=this.min();
            hi=this.max();
          }
        }
        if(!this.isEmpty()&& this._argOk(lo) && this._argOk(hi)){
          this._keysNode(this.root, Q, lo, hi);
        }
        return Q.iter();
      }
      // add the keys between lo and hi in the subtree rooted at x
      // to the queue
      _keysNode(x, queue, lo, hi){
        if(x){
          let cmplo = this.compare(lo,x.key);
          let cmphi = this.compare(hi,x.key);
          if(cmplo < 0) this._keysNode(x.left, queue, lo, hi);
          if(cmplo <= 0 && cmphi >= 0) queue.enqueue(x.key);
          if(cmphi > 0) this._keysNode(x.right, queue, lo, hi);
        }
        return queue;
      }
      /**Returns the number of keys in the symbol table in the given range.
       * @param  {any} lo minimum endpoint
       * @param  {any} hi maximum endpoint
       * @return {number} the number of keys in the symbol table between {@code lo}
       *    (inclusive) and {@code hi} (inclusive)
       */
      size(lo, hi){
        return arguments.length==0? this._sizeNode(this.root)
          : (this._argOk(lo) &&
             this._argOk(hi) &&
             this.compare(lo,hi) > 0? 0
                                    :(this.contains(hi)? (this.rank(hi) - this.rank(lo) + 1)
                                                       : (this.rank(hi) - this.rank(lo))));
      }
      static load(input,compareFn){
        let b= new RedBlackBST(compareFn);
        input.forEach((s,i)=> b.put(s,i));
        return b;
      }
      static test(){
        let m,obj= RedBlackBST.load("SEARCHEXAMPLE".split(""), CMP);
        m="";
        for(let s,it=obj.keys();it.hasNext();){
          s=it.next(); m+=`${s}=${obj.get(s)} `; }
        console.log(m);
        obj.isEmpty();
        console.log("height= "+obj.height()+", size= "+obj.size());
        console.log("get X= "+obj.get("X"));
        console.log("contains X= "+obj.contains("X"));
        console.log("min= "+obj.min()+",max= "+obj.max());
        obj.deleteMin();
        obj.deleteMax();
        console.log("min= "+obj.min()+",max= "+obj.max());
        obj.delete("R");
        console.log("contains R= "+obj.contains("R"));
        console.log("floor J= "+obj.floor("J"));
        console.log("ceiling J= "+obj.ceiling("J"));
        console.log("rank M= "+obj.rank("M"));
        m="";
        for(let s,it=obj.keys("D","Q");it.hasNext();){
          s=it.next(); m+=`${s}=${obj.get(s)} `; }
        console.log("keys[D-Q]= "+m);
        console.log("size[E-P]= "+ obj.size("E","P"));
      }
    }
    //RedBlackBST.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides a static method for binary searching for an integer in a sorted array of integers.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class BinarySearch{
      /**Returns the index of the specified key in the specified array.
       * @param  {array} a the array of integers, must be sorted in ascending order
       * @param  {number} key the search key
       * @return {number} index of key in array {@code a} if present; {@code -1} otherwise
       */
      static indexOf(a, key){
        let lo = 0,
            hi = a.length - 1;
        while(lo <= hi){
          // Key is in a[lo..hi] or not present.
          let mid = lo + _M.ndiv(hi-lo,2);
          if(key < a[mid]) hi = mid - 1;
          else if(key > a[mid]) lo = mid + 1;
          else return mid;
        }
        return -1;
      }
      static test(){
        let inp= "84 48 68 10 18 98 12 23 54 57 33 16 77 11 29".split(" ").map(s=>{ return +s }).sort();
        let t="23 50 10 99 18 23 98 84 11 10 48 77 13 54 98 77 77 68".split(" ").map(s=>{return +s});
        t.forEach(n=>{
          if(BinarySearch.indexOf(inp,n)<0)
            console.log(n);
        })
      }
    }
    //BinarySearch.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an ordered symbol table of generic key-value pairs.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class AVLTreeST{
      Node(key, val, height, size){
        // height: height of the subtree
        // size: number of nodes in subtree
        return {key, val, height, size};// left:null, right: null
      }
      /**
       * @param {function} compareFn
       */
      constructor(compareFn){
        this.compare=compareFn;
        this.root=UNDEF;
      }
      /**Checks if the symbol table is empty.
       * @return {boolean}
       */
      isEmpty(){
        return _.nichts(this.root)
      }
      //Returns the number of nodes in the subtree.
      _sizeNode(x){
        return _.nichts(x) ? 0:x.size;
      }
      /**Returns the height of the internal AVL tree. It is assumed that the
       * height of an empty tree is -1 and the height of a tree with just one node
       * is 0.
       * @return {number}
       */
      height(){
        return this._heightNode(this.root);
      }
      //Returns the height of the subtree.
      _heightNode(x){
        return _.nichts(x)? -1: x.height;
      }
      /**Returns the value associated with the given key.
       * @param {any} key the key
       * @return {any} the value associated with the given key if the key is in the
       *         symbol table and {@code null} if the key is not in the
       *         symbol table
       */
      get(key){
        if(_.nichts(key)) throw Error("argument to get() is null");
        let x = this._getNode(this.root, key);
        if(x) return x.val;
      }
      /**Returns value associated with the given key in the subtree or
       * {@code null} if no such key.
       *
       * @param {object} x the subtree
       * @param {any} key the key
       * @return {any} value associated with the given key in the subtree or
       *         {@code null} if no such key
       */
      _getNode(x, key){
        if(!x) return UNDEF;
        let cmp = this.compare(key,x.key);
        if(cmp < 0) return this._getNode(x.left, key);
        if(cmp > 0) return this._getNode(x.right, key);
        return x;
      }
      /**Checks if the symbol table contains the given key.
       * @param {any} key the key
       * @return {boolean}
       */
      contains(key){
        return this.get(key) !== undefined;
      }
      /**Inserts the specified key-value pair into the symbol table, overwriting
       * the old value with the new value if the symbol table already contains the
       * specified key. Deletes the specified key (and its associated value) from
       * this symbol table if the specified value is {@code null}.
       * @param {any} key the key
       * @param {any} val the value
       */
      put(key, val){
        if(_.nichts(key)) throw Error("first argument to put() is null");
        if(val === undefined){
          this.delete(key);
        }else{
          this.root = this._putNode(this.root, key, val);
        }
      }
      /**Inserts the key-value pair in the subtree. It overrides the old value
       * with the new value if the symbol table already contains the specified key
       * and deletes the specified key (and its associated value) from this symbol
       * table if the specified value is {@code null}.
       * @param {object} x the subtree
       * @param {any} key the key
       * @param {any} val the value
       * @return {object} the subtree
       */
      _putNode(x, key, val){
        if(!x) return this.Node(key, val, 0, 1);
        let cmp = this.compare(key,x.key);
        if(cmp < 0){
          x.left = this._putNode(x.left, key, val);
        }else if(cmp > 0){
          x.right = this._putNode(x.right, key, val);
        }else{
          x.val = val;
          return x;
        }
        x.size = 1 + this._sizeNode(x.left) + this._sizeNode(x.right);
        x.height = 1 + Math.max(this._heightNode(x.left), this._heightNode(x.right));
        return this._balanceNode(x);
      }
      /**Restores the AVL tree property of the subtree.
       * @param {object} x the subtree
       * @return {object} the subtree with restored AVL property
       */
      _balanceNode(x){
        if(this._balanceFactor(x) < -1){
          if(this._balanceFactor(x.right) > 0) x.right = this._rotateRight(x.right);
          x = this._rotateLeft(x);
        }else if(this._balanceFactor(x) > 1){
          if(this._balanceFactor(x.left) < 0) x.left = this._rotateLeft(x.left);
          x = this._rotateRight(x);
        }
        return x;
      }
      /**Returns the balance factor of the subtree. The balance factor is defined
       * as the difference in height of the left subtree and right subtree, in
       * this order. Therefore, a subtree with a balance factor of -1, 0 or 1 has
       * the AVL property since the heights of the two child subtrees differ by at
       * most one.
       * @param {object} x the subtree
       * @return {number} the balance factor of the subtree
       */
      _balanceFactor(x){
        return this._heightNode(x.left) - this._heightNode(x.right);
      }
      /** Rotates the given subtree to the right.
       * @param {object} x the subtree
       * @return {object} the right rotated subtree
       */
      _rotateRight(x){
        let y = x.left;
        x.left = y.right;
        y.right = x;
        y.size = x.size;
        x.size = 1 + this._sizeNode(x.left) + this._sizeNode(x.right);
        x.height = 1 + Math.max(this._heightNode(x.left), this._heightNode(x.right));
        y.height = 1 + Math.max(this._heightNode(y.left), this._heightNode(y.right));
        return y;
      }
      /**Rotates the given subtree to the left.
       * @param {object} x the subtree
       * @return {oject} the left rotated subtree
       */
      _rotateLeft(x){
        let y = x.right;
        x.right = y.left;
        y.left = x;
        y.size = x.size;
        x.size = 1 + this._sizeNode(x.left) + this._sizeNode(x.right);
        x.height = 1 + Math.max(this._heightNode(x.left), this._heightNode(x.right));
        y.height = 1 + Math.max(this._heightNode(y.left), this._heightNode(y.right));
        return y;
      }
      /**Removes the specified key and its associated value from the symbol table
       * (if the key is in the symbol table).
       * @param {any} key the key
       */
      delete(key){
        if(_.nichts(key)) throw Error("argument to delete() is null");
        if(this.contains(key))
          this.root = this._deleteNode(this.root, key);
      }
      /**Removes the specified key and its associated value from the given subtree.
       * @param {object} x the subtree
       * @param {any} key the key
       * @return {object} the updated subtree
       */
      _deleteNode(x, key){
        let cmp = this.compare(key,x.key);
        if(cmp < 0){
          x.left = this._deleteNode(x.left, key);
        }else if(cmp > 0){
          x.right = this._deleteNode(x.right, key);
        }else{
          if(!x.left) return x.right;
          if(!x.right) return x.left;
          let y = x;
          x = this.min(y.right);
          x.right = this.deleteMin(y.right);
          x.left = y.left;
        }
        x.size = 1 + this._sizeNode(x.left) + this._sizeNode(x.right);
        x.height = 1 + Math.max(this._heightNode(x.left), this._heightNode(x.right));
        return this._balance(x);
      }
      /**Removes the smallest key and associated value from the symbol table.
       */
      deleteMin(){
        if(this.isEmpty()) throw Error("called deleteMin() with empty symbol table");
        this.root = this._deleteMinNode(this.root);
      }
      /**Removes the smallest key and associated value from the given subtree.
       * @param {object} x the subtree
       * @return {object} the updated subtree
       */
      _deleteMinNode(x){
        if(!x.left) return x.right;
        x.left = this._deleteMinNode(x.left);
        x.size = 1 + this._sizeNode(x.left) + this._sizeNode(x.right);
        x.height = 1 + Math.max(this._heightNode(x.left), this._heightNode(x.right));
        return this._balance(x);
      }
      /**Removes the largest key and associated value from the symbol table.
       */
      deleteMax(){
        if(this.isEmpty()) throw Error("called deleteMax() with empty symbol table");
        this.root = this._deleteMaxNode(this.root);
      }
      /**Removes the largest key and associated value from the given subtree.
       * @param {object} x the subtree
       * @return {object} the updated subtree
       */
      _deleteMaxNode(x){
        if(!x.right) return x.left;
        x.right = this._deleteMaxNode(x.right);
        x.size = 1 + this._sizeNode(x.left) + this._sizeNode(x.right);
        x.height = 1 + Math.max(this._heightNode(x.left), this._heightNode(x.right));
        return this._balance(x);
      }
      /**Returns the smallest key in the symbol table.
       * @return {any} the smallest key in the symbol table
       */
      min(){
        if(this.isEmpty()) throw Error("called min() with empty symbol table");
        return this._minNode(this.root).key;
      }
      /**Returns the node with the smallest key in the subtree.
       * @param {object} x the subtree
       * @return {object} the node with the smallest key in the subtree
       */
      _minNode(x){
        return !x.left ? x: this._minNode(x.left);
      }
      /**Returns the largest key in the symbol table.
       * @return the largest key in the symbol table
       */
      max(){
        if(this.isEmpty()) throw Error("called max() with empty symbol table");
        return this._maxNode(this.root).key;
      }
      /**Returns the node with the largest key in the subtree.
       * @param {object} x the subtree
       * @return {object} the node with the largest key in the subtree
       */
      _maxNode(x){
        return !x.right ? x: this._maxNode(x.right);
      }
      /**Returns the largest key in the symbol table less than or equal to
       * {@code key}.
       * @param {any} key the key
       * @return {any} the largest key in the symbol table less than or equal to
       *         {@code key}
       */
      floor(key){
        if(_.nichts(key)) throw Error("argument to floor() is null");
        if(this.isEmpty()) throw Error("called floor() with empty symbol table");
        let x = this._floorNode(this.root, key);
        if(x) return x.key;
      }
      /**Returns the node in the subtree with the largest key less than or equal
       * to the given key.
       * @param {object} x the subtree
       * @param {any} key the key
       * @return {object} the node in the subtree with the largest key less than or equal
       *         to the given key
       */
      _floorNode(x, key){
        if(_.nichts(x)) return UNDEF;
        let cmp = this.compare(key,x.key);
        if(cmp == 0) return x;
        if(cmp < 0) return this._floorNode(x.left, key);
        let y = this._floorNode(x.right, key);
        return y?  y : x;
      }
      /**Returns the smallest key in the symbol table greater than or equal to
       * {@code key}.
       * @param {any} key the key
       * @return {any} the smallest key in the symbol table greater than or equal to
       *         {@code key}
       */
      ceiling(key){
        if(_.nichts(key)) throw Error("argument to ceiling() is null");
        if(this.isEmpty()) throw Error("called ceiling() with empty symbol table");
        let x = this._ceilingNode(this.root, key);
        if(x) return x.key;
      }
      /**Returns the node in the subtree with the smallest key greater than or
       * equal to the given key.
       * @param {object} x the subtree
       * @param {any} key the key
       * @return {object} the node in the subtree with the smallest key greater than or
       *         equal to the given key
       */
      _ceilingNode(x, key){
        if(_.nichts(x)) return UNDEF;
        let cmp = this.compare(key,x.key);
        if(cmp == 0) return x;
        if(cmp > 0) return this._ceilingNode(x.right, key);
        let y = this._ceilingNode(x.left, key);
        return y? y: x;
      }
      /**Returns the kth smallest key in the symbol table.
       * @param {number} k the order statistic
       * @return {any} the kth smallest key in the symbol table
       */
      select(k){
        if(k < 0 || k >= this.size()) throw Error("k is not in range 0-" + (this.size() - 1));
        let n= this._selectNode(this.root, k);
        if(n) return n.key;
      }
      /**Returns the node with key the kth smallest key in the subtree.
       * @param {object} x the subtree
       * @param {any} k the kth smallest key in the subtree
       * @return {object} the node with key the kth smallest key in the subtree
       */
      _selectNode(x, k){
        if(_.nichts(x)) return UNDEF;
        let t = this._sizeNode(x.left);
        if(t > k) return this._selectNode(x.left, k);
        if(t < k) return this._selectNode(x.right, k - t - 1);
        return x;
      }
      /**Returns the number of keys in the symbol table strictly less than
       * {@code key}.
       * @param {any} key the key
       * @return {number} the number of keys in the symbol table strictly less than
       *         {@code key}
       */
      rank(key){
        if(_.nichts(key)) throw Error("argument to rank() is null");
        return this._rankNode(key, this.root);
      }
      /**Returns the number of keys in the subtree less than key.
       * @param {any} key the key
       * @param {object} x the subtree
       * @return {number} the number of keys in the subtree less than key
       */
      _rankNode(key, x){
        if(_.nichts(x)) return 0;
        let cmp = this.compare(key,x.key);
        if(cmp < 0) return this._rankNode(key, x.left);
        if(cmp > 0) return 1 + this._sizeNode(x.left) + this._rankNode(key, x.right);
        return this._sizeNode(x.left);
      }
      /**Returns all keys in the symbol table following an in-order traversal.
       * @return {Iterator} all keys in the symbol table following an in-order traversal
       */
      keysInOrder(){
        let queue = new Queue();
        this._keysInOrderNode(this.root, queue);
        return queue.iter();
      }
      /**Adds the keys in the subtree to queue following an in-order traversal.
       * @param {object} x the subtree
       * @param {Queue} queue the queue
       */
      _keysInOrderNode(x, queue){
        if(!_.nichts(x)){
          this._keysInOrderNode(x.left, queue);
          queue.enqueue(x.key);
          this._keysInOrderNode(x.right, queue);
        }
      }
      /**Returns all keys in the symbol table following a level-order traversal.
       * @return {Iterator} all keys in the symbol table following a level-order traversal.
       */
      keysLevelOrder(){
        let queue = new Queue();
        if(!this.isEmpty()){
          let queue2 = new Queue();
          queue2.enqueue(this.root);
          while(!queue2.isEmpty()){
            let x = queue2.dequeue();
            queue.enqueue(x.key);
            if(!x.left ){
              queue2.enqueue(x.left);
            }
            if(x.right ){
              queue2.enqueue(x.right);
            }
          }
        }
        return queue;
      }
      /**Returns all keys in the symbol table in the given range.
       * @param {any} lo the lowest key
       * @param {any} hi the highest key
       * @return {Iterator} all keys in the symbol table between {@code lo} (inclusive)
       *         and {@code hi} (exclusive)
       */
      keys(lo, hi){
        if(arguments.length==0){ return this.keysInOrder()}
        if(_.nichts(lo)) throw Error("first argument to keys() is null");
        if(_.nichts(hi)) throw Error("second argument to keys() is null");
        let queue = new Queue();
        this._keysNode(this.root, queue, lo, hi);
        return queue.iter();
      }
      /**Adds the keys between {@code lo} and {@code hi} in the subtree
       * to the {@code queue}.
       */
      _keysNode(x, queue, lo, hi){
        if(x){
          let cmplo = this.compare(lo,x.key);
          let cmphi = this.compare(hi,x.key);
          if(cmplo < 0) this._keysNode(x.left, queue, lo, hi);
          if(cmplo <= 0 && cmphi >= 0) queue.enqueue(x.key);
          if(cmphi > 0) this._keysNode(x.right, queue, lo, hi);
        }
      }
      /**Returns the number of keys in the symbol table in the given range.
       * @param lo minimum endpoint
       * @param hi maximum endpoint
       * @return the number of keys in the symbol table between {@code lo}
       *         (inclusive) and {@code hi} (exclusive)
       * @throws Error if either {@code lo} or {@code hi} is {@code null}
       */
      size(lo, hi){
        if(arguments.length==0){ return this._sizeNode(this.root)}
        if(_.nichts(lo)) throw Error("first argument to size() is null");
        if(_.nichts(hi)) throw Error("second argument to size() is null");
        if(this.compare(lo,hi) > 0) return 0;
        if(this.contains(hi)) return this.rank(hi) - this.rank(lo) + 1;
        return this.rank(hi) - this.rank(lo);
      }
      //Checks if the AVL tree invariants are fine.
      _check(){
        let self=this;
        function isAVL(x){
          if(!x) return true;
          let bf = self._balanceFactor(x);
          if(bf > 1 || bf < -1) return false;
          return isAVL(x.left) && isAVL(x.right);
        }
        function isBST(x, min, max){
          if(!x) return true;
          if(!min && self.compare(x.key,min) <= 0) return false;
          if(!max && self.compare(x.key,max) >= 0) return false;
          return isBST(x.left, min, x.key) && isBST(x.right, x.key, max);
        }
        function isSizeConsistent(x){
          if(!x) return true;
          if(x.size != self._sizeNode(x.left) + self._sizeNode(x.right) + 1) return false;
          return isSizeConsistent(x.left) && isSizeConsistent(x.right);
        }
        function isRankConsistent(){
          for(let i = 0; i < self.size(); i++)
            if(i != self.rank(self.select(i))) return false;
          for(let k, it=self.keys().iterator();it.hasNext();){
            k=it.next();
            if(this.compare(k,self.select(self.rank(key))) != 0) return false;
          }
          return true;
        }
        return isBST(this.root,null,null) && isAVL(this.root) && isSizeConsistent(this.root) && isRankConsistent();
      }
      static test(){
        let st = new AVLTreeST(CMP);
        "SEARCHEXAMPLE".split("").forEach((s,i)=> st.put(s,i));
        for(let s,it=st.keys();it.hasNext();){
          s=it.next();
          console.log(s + " " + st.get(s));
        }
      }
    }
    //AVLTreeST.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const SQRT2=Math.sqrt(2);
    function AStarGridNode(loc,par){
      return{
        parent: par, pos: loc, f:0, g:0, h:0,
        pid: `${loc[0]},${loc[1]}`,
        equals(o){
          return this.pos[0]==o.pos[0] &&
                 this.pos[1]==o.pos[1]
        }
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**A* search algo for grid.
     * @memberof module:mcfud/algo_search
     * @class
     */
    class AStarGrid{
      /**Use when movement is limited to 4 directions only.
       * @param {array} test node pos
       * @param {array} goal node pos
       * @param {number} cost
       */
      static manhattan(test, goal,cost=1){
        return cost*Math.abs(test[1] - goal[1]) +
               cost*Math.abs(test[0] - goal[0]);
      }
      /**Use when movements are allowed in all directions.
       * @param {array} test node pos
       * @param {array} goal node pos
       * @param {number} cost
       */
      static euclidean(test, goal,cost=1){
        let vx = goal[0] - test[0],
            vy = goal[1] - test[1];
        return cost * (vx * vx + vy * vy);
      }
      /**Use when movements are allowed in all directions.
       * @param {array} test node pos
       * @param {array} goal node pos
       * @param {number} cost
       */
      static diagonal(test, goal,cost=1,xcost=SQRT2){
        let dx = Math.abs(goal[0] - test[0]),
            dy = Math.abs(goal[1] - test[1]);
        return cost * (dx + dy) + (xcost - 2 * cost) * Math.min(dx, dy);
      }
      constructor(grid){
        this.grid=grid;
      }
      pathTo(start, end, ctx){
        return this._search(this.grid,start,end,ctx)
      }
      _search(grid,start,end,ctx){
        const CMP=ctx.compare,
              ROWS= grid.length,
              COLS= grid[0].length,
              closedSet = new Map(),
              openTM= new Map(),
              openSet = new MinPQ(CMP,10),
              goalNode = AStarGridNode(end),
              startNode = AStarGridNode(start),
              dirs=[[1,0],[-1,0],[0,1],[0,-1]],
              rpath=(cn,out)=>{ for(;cn;cn=cn.parent) out.unshift(cn.pos); return out; };
        //include diagonal neighbors?
        if(ctx.wantDiagonal)
          dirs.push([1,1],[1,-1],[-1,1],[-1,-1]);
        openTM.set(startNode.pid,startNode.g);
        openSet.insert(startNode);
        //begin...
        let cur,neighbors=[];
        while(!openSet.isEmpty()){
          cur= openSet.delMin();
          openTM.delete(cur.pid);
          closedSet.set(cur.pid,0);
          //done?
          if(cur.equals(goalNode)){return rpath(cur,[])}
          neighbors.length=0;
          for(let p,i=0;i<dirs.length;++i){
            p = [cur.pos[0] + dirs[i][0], cur.pos[1] + dirs[i][1]];
            if(p[0] > (COLS-1) || p[0] < 0 ||
               p[1] > (ROWS-1) || p[1] < 0 || ctx.blocked(p)){
            }else{
              neighbors.push(AStarGridNode(p,cur));
            }
          }
          neighbors.forEach(co=>{
            if(!closedSet.has(co.pid)){
              co.g = cur.g + ctx.cost();
              co.h = ctx.calcHeuristic(co.pos,goalNode.pos);
              co.f = co.g + co.h;
              //update if lower cost
              if(openTM.has(co.pid) && co.g > openTM.get(co.pid)){}else{
                openSet.insert(co);
                openTM.set(co.pid, co.g);
              }
            }
          });
        }
      }
      static test(){
        let grid = [[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0]];
        let ROWS=grid.length,COLS=grid[0].length;
        let ctx={
          wantDiagonal:false,
          compare(a,b){ return a.f-b.f },
          cost(){ return 1 },
          blocked(n){ return grid[n[1]][n[0]] != 0 },
          calcHeuristic(a,g){
            //return AStarGrid.diagonal(a,g,10,14);
            return AStarGrid.euclidean(a,g);
            //return AStarGrid.manhattan(a,g,10)
          }
        }
        let c,r,m,p= new AStarGrid(grid).pathTo([0,0],[5,4],ctx);
        if(p){
          m=""; p.forEach(n=>{ m+= `[${n[0]},${n[1]}] `; }); console.log(m);
          r=_.fill(ROWS, ()=> _.fill(COLS, "#"));
          c=0;
          p.forEach(n=>{
            r[n[1]][n[0]]= ""+c;
            ++c;
          });
          r.forEach(row=>{
            console.log(row.toString())
          });
        }else{
          console.log("no path");
        }
      }
    }
    //AStarGrid.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      AStarGrid,
      AVLTreeST,
      RedBlackBST,
      BST,
      BinarySearch,
      BinarySearchST,
      FrequencyCounter,
      SequentialSearchST
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"), require("./sort"))
  }else{
    gscope["io/czlab/mcfud/algo/search"]=_module
  }

})(this);


