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
// Copyright © 2013-2022, Kenneth Leung. All rights reserved.

;(function(gscope,UNDEF){

  "use strict";

  /**Create the module.
   */
  function _module(Core){

    if(!Core) Core= gscope["io/czlab/mcfud/core"]();

    const CMP=(a,b)=> a<b?-1:(a>b?1:0);
    const int=Math.floor;
    const {is, u:_}= Core;

    /**
     * @module mcfud/algo_basic
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _checkKey=(key)=> _.assert(is.num(key) || is.str(key), `expected number or string`);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function prnIter(it, sep=" ",out=""){
      for(; it.hasNext();) out += `${it.next()}${sep}`;
      return out;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an interable.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class Iterator{
      constructor(c){
        this.current=c;// node to starts with
      }
      hasNext(){ return _.echt(this.current) }
      remove(){ throw Error("Unsupported")  }
      next(){
        if(!this.hasNext())
          throw Error("NoSuchElementException");
        let item = this.current.item;
        this.current = this.current.next;
        return item;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Helper linked list
     * @memberof module:mcfud/algo_basic
     * @param {any} item
     * @param {object} next
     * @return {object}
     */
    function Node(item,next=UNDEF){
      return { item, next }
    }

    /**Represents a bag (or multiset) of generic items.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class Bag{
      constructor(){
        //* @property {first} beginning of bag
        //* @property {n} number of elements in bag
        this.first = UNDEF;
        this.n = 0;
      }
      clone(){
        let b=new Bag(),
            q,p=this.first;
        while(p){
          if(q){
            q.next=Node(p.item);
            q=q.next;
          }else{
            b.first= q =Node(p.item);
          }
          b.n+=1;
          p=p.next;
        }
        return b;
      }
      /**Returns true if this bag is empty.
       * @return {boolean}
       */
      isEmpty(){
        return _.nichts(this.first)
      }
      /**Returns the number of items in this bag.
       * @return {number}
       */
      size(){
        return this.n;
      }
      /**Adds the item to this bag.
       * @param {any} item the item to add to this bag
       */
      add(item){
        //adds this back to front, first always points to last added
        this.first = Node(item, this.first);
        this.n+=1;
      }
      /**Returns an iterator that iterates over the items in this bag in arbitrary order.
       * @return {Iterator}
       */
      iter(){
        return new Iterator(this.first);
      }
      static test(){
        let obj= new Bag();
        "to be or not to - be - - that - - - is".split(" ").forEach(n=> obj.add(n));
        console.log("size of bag = " + obj.size());
        console.log(prnIter(obj.iter()));
        let c=obj.clone();
        console.log("size of cloned = " + c.size());
        console.log(prnIter(c.iter()));
      }
    }
    //Bag.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a last-in-first-out (LIFO) stack of generic items.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class Stack{
      constructor(){
        //* @property {first} top of stack
        //* @property {n} size of stack
        this.first = UNDEF;
        this.n = 0;
      }
      clone(){
        let b=new Stack(),
            q,p=this.first;
        while(p){
          if(q){
            q.next=Node(p.item);
            q=q.next;
          }else{
            b.first= q =Node(p.item);
          }
          b.n+=1;
          p=p.next;
        }
        return b;
      }
      /**Returns true if this stack is empty.
       * @return {boolean}
       */
      isEmpty(){
        return _.nichts(this.first);
      }
      /**Returns the number of items in this stack.
       * @return {number}
       */
      size(){
        return this.n;
      }
      /**Adds the item to this stack.
       * @param {any} item
       */
      push(item){
        this.first = Node(item, this.first);
        this.n+=1;
      }
      /**Removes and returns the item most recently added to this stack.
       * @return {any}
       */
      pop(){
        if(this.isEmpty())
          throw Error("Stack underflow");
        let item = this.first.item; // save item to return
        this.first = this.first.next; // delete first node
        this.n -= 1;
        return item;                   // return the saved item
      }
      /**Returns (but does not remove) the item most recently added to this stack.
       * @return {any}
       */
      peek(){
        if(this.isEmpty())
          throw Error("Stack underflow");
        return this.first.item;
      }
      /**Returns a string representation of this stack.
       * @return {string}
       */
      toString(){
        return prnIter(this.iter())
      }
      /**Returns an iterator to this stack that iterates through the items in LIFO order.
       * @return {Iterator}
       */
      iter(){
        return new Iterator(this.first)
      }
      static test(){
        let obj= new Stack();
        "to be or not to - be - - that - - - is".split(" ").forEach(s=>{
          if(s != "-"){
            obj.push(s);
          }else if(!obj.isEmpty()){
            console.log("(-)" + obj.pop() + " ");
          }
        });
        console.log("(" + obj.size() + " left on stack)");
        let c= obj.clone();
        console.log("cloned= " + prnIter(c.iter()));
        console.log("(" + c.size() + " left on stack)");
      }
    }
    //Stack.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a first-in-first-out (FIFO) queue of generic items.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class Queue{
      constructor(){
        //* @property {first} beginning of queue
        //* @property {last} end of queue
        //* @property {n} number of elements on queue
        this.first = UNDEF;
        this.last  = UNDEF;
        this.n = 0;
      }
      clone(){
        let q=new Queue(),
            p=this.first;
        while(p){
          q.enqueue(p.item);
          p=p.next;
        }
        return q;
      }
      /**Returns true if this queue is empty.
       * @return {boolean}
       */
      isEmpty(){
        return _.nichts(this.first)
      }
      /**Returns the number of items in this queue.
       * @return {number}
       */
      size(){
        return this.n;
      }
      /**Returns the item least recently added to this queue.
       * @return {any}
       */
      peek(){
        if(this.isEmpty()) throw Error("Queue underflow");
        return this.first.item;
      }
      /**Adds the item to this queue.
       * @param {any} item
       */
      enqueue(item){
        let oldlast = this.last;
        this.last = Node(item);
        if(this.isEmpty()) this.first = this.last;
        else oldlast.next = this.last;
        this.n+=1;
      }
      /**Removes and returns the item on this queue that was least recently added.
       * @return {any}
       */
      dequeue(){
        if(this.isEmpty()) throw Error("Queue underflow");
        let item = this.first.item;
        this.first = this.first.next;
        this.n-=1;
        if(this.isEmpty()) this.last = UNDEF;   // to avoid loitering
        return item;
      }
      /**Returns a string representation of this queue.
       * @return {string}
       */
      toString(){
        return prnIter(this.iter())
      }
      /**Returns an iterator that iterates over the items in this queue in FIFO order.
       * @return {Iterator}
       */
      iter(){
        return new Iterator(this.first)
      }
      static test(){
        let queue = new Queue();
        "to be or not to - be - - that - - - is".split(/\s+/).forEach(s=>{
          if(s!="-")
            queue.enqueue(s);
          else if(!queue.isEmpty())
            console.log(queue.dequeue() + " ");
        });
        console.log("(" + queue.size() + " left on queue)");
        let c= queue.clone();
        console.log("cloned= "+ prnIter(c.iter()));
        console.log("(" + c.size() + " left on queue)");
      }
    }
    //Queue.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //Original source from https://github.com/awstuff/TreeMap.js
    /**The map is sorted according to the natural ordering of its keys,or via the
     * provided comparator.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class TreeMap{
      /**
       * @param {function} C compare function
       */
      constructor(C){
        //* @property {function} compare comparator function
        //* @property {object} root top of the map
        //* @property {number} n number of elements in map
        this.compare=C || CMP;
        this.root=UNDEF;
        this.n=0;
      }
      /**Number of entries in map.
       * @return {number} size of map
       */
      size(){
        return this.n
      }
      /**Check if key exists in map..
       * @param {any} key
       * @return {boolean}
       */
      contains(key){
        return this.get(key) !== undefined
      }
      /**Get value for this key.
       * @param {any} key
       * @return {any} value if found else undefined
       */
      get(key){
        const _get=(key,node,res)=>{
          if(node){
            let c=this.compare(key,node.key);
            res= c<0 ?_get(key, node.left) : (c>0 ? _get(key, node.right): node.value);
          }
          return res;
        }
        if(_checkKey(key) && this.n>0) return _get(key,this.root);
      }
      /**Associate key with this value
       * @param {any} key
       * @param {any} val cannot be undefined
       */
      set(key, value){
        const _set=(key,value,node)=>{
          if(!node){
            this.n+=1;
            return{key,value};
          }
          let c= this.compare(key,node.key);
          if(c<0){
            node.left = _set(key, value, node.left);
          }else if(c>0){
            node.right = _set(key, value, node.right);
          }else{
            node.value = value;
          }
          return node;
        }
        if(_checkKey(key) &&
           value !== undefined)
          this.root = _set(key, value, this.root);
      }
      _getMaxNode(node){
        while(node && node.right){ node = node.right }
        return node;
      }
      _getMaxKey(){
        let n = this._getMaxNode(this.root);
        if(n)
          return n.key;
      }
      _getMinNode(node){
        while(node && node.left){ node = node.left }
        return node;
      }
      _getMinKey(){
        let n = this._getMinNode(this.root);
        if(n)
          return n.key;
      }
      /**Remove key from map.
       * @param {any} key
       */
      remove(key){
        const _del=(key,node)=>{
          if(node){
            let k,v,m,c= this.compare(key,node.key);
            if(c<0){
              node.left = _del(key, node.left);
            }else if(c>0){
              node.right = _del(key, node.right);
            }else{
              if(node.left && node.right){
                m = this._getMaxNode(node.left);
                k = m.key;
                v = m.value;
                m.value = node.value;
                m.key = node.key;
                node.key = k;
                node.value = v;
                node.left = _del(key, node.left);
              }else if(node.left){
                node=node.left;
                this.n -=1;
              }else if(node.right){
                node= node.right;
                this.n -=1;
              }else{
                node= UNDEF;
                this.n -=1;
              }
            }
          }
          return node;
        }
        if(_checkKey(key))
          this.root = _del(key, this.root);
      }
      /**List keys in this map.
       * @return {Iterator}
       */
      keys(){
        let out=new Queue();
        this.forEach((v,k)=> out.enqueue(k));
        return out.iter();
      }
      /**Get the first key.
       * @return {any}
       */
      firstKey(){
        let ret;
        try{
          this.forEach((v,k)=>{
            ret=k;
            throw Error("????");
          });
        }catch(e){}
        return ret;
      }
      /**Get the last key
       * @return {any} the last key
       */
      lastKey(){
        let ret;
        this.forEach((v,k)=>{ ret=k });
        return ret;
      }
      /**Iterate the map.
       * @param {function} cb callback function
       * @param {object} ctx callback context
       */
      forEach(cb, ctx){
        function _invokeCb(ctx,cb){
          return cb && cb.apply(ctx, Array.prototype.slice.call(arguments, 2)) }
        function _each(node, cb, ctx, func){
          if(!node)
            return _invokeCb(ctx, func);
          _each(node.left, cb, ctx, function(){
            _invokeCb(ctx, cb, node.value, node.key);
            _each(node.right, cb, ctx, function(){ _invokeCb(ctx, func) });
          });
        }
        _each(this.root, cb, ctx);
      }
      static test(){
        let t= new TreeMap();
        t.set(3, "3"); t.set(2,"2"); t.set(7,"7"), t.set(1,"1");
        console.log(`firstKey= ${t.firstKey()}`);
        console.log(`lastKey= ${t.lastKey()}`);
        console.log(prnIter(t.keys()));
        console.log(`k= ${t.get(3)}`);
        console.log(`has 2 = ${t.contains(2)}`);
        console.log(`has size = ${t.size()}`);
        t.remove(1);
        console.log(`has size = ${t.size()}`);
        console.log(prnIter(t.keys()));
        console.log(`k= ${t.get(2)}`);
        //t= new TreeMap();
        //t.set("ghi", "3"); t.set("def","2"); t.set("jkl","7"), t.set("zbc","1");
        //console.log(prnIter(t.keys()));
      }
    }
    //TreeMap.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an ordered symbol table of generic key-value pairs.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class ST{
      constructor(){
        this.st = new TreeMap();
      }
      /**Returns the value associated with the given key in this symbol table.
       * @param  key the key
       * @return {any}
       */
      get(key){
        if(_.nichts(key)) throw Error("calls get() with null key");
        return this.st.get(key);
      }
      /**Inserts the specified key-value pair into the symbol table, overwriting the old
       * value with the new value if the symbol table already contains the specified key.
       * Deletes the specified key (and its associated value) from this symbol table
       * if the specified value is {@code null}.
       * @param  {any} key the key
       * @param  {any} val the value, if undefined, key is removed
       */
      put(key, val){
        if(_.nichts(key)) throw Error("calls put() with null key");
        if(val === undefined) this.st.remove(key);
        else this.st.set(key, val);
      }
      /**Removes the specified key and its associated value from this symbol table
       * (if the key is in this symbol table).
       * This is equivalent to {@code remove()}, but we plan to deprecate {@code delete()}.
       * @param  key the key
       */
      remove(key){
        if(_.nichts(key)) throw Error("calls remove() with null key");
        this.st.remove(key);
      }
      /**Returns true if this symbol table contain the given key.
       * @param  {any} key
       * @return {boolean}
       */
      contains(key){
        if(_.nichts(key)) throw Error("calls contains() with null key");
        return this.st.contains(key);
      }
      /**Returns the number of key-value pairs in this symbol table.
       * @return {number}
       */
      size(){
        return this.st.size();
      }
      /**Returns true if this symbol table is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.size() == 0;
      }
      /**Returns all keys in this symbol table.
       * To iterate over all of the keys in the symbol table named {@code st},
       * use the foreach notation: {@code for (Key key : st.keys())}.
       * @return {Iterator}
       */
      keys(){
        return this.st.keys();
      }
      /**Returns the smallest key in this symbol table.
       * @return {any}
       */
      min(){
        if(this.isEmpty()) throw Error("calls min() with empty symbol table");
        return this.st.firstKey();
      }
      /**Returns the largest key in this symbol table.
       * @return {any}
       */
      max(){
        if(this.isEmpty()) throw Error("calls max() with empty symbol table");
        return this.st.lastKey();
      }
      /**Returns the smallest key in this symbol table greater than or equal to {@code key}.
       * @param  {any} key
       * @return {any}
       */
      ceiling(key){
        if(_.nichts(key)) throw Error("argument to ceiling() is null");
        let w,k,it= this.st.keys();
        while(it.hasNext()){
          k=it.next();
          if(k == key || k>key){
            w=k;
            break;
          }
        }
        if(w===undefined)
          throw Error("argument to ceiling() is too large");
        return w;
      }
      /**Returns the largest key in this symbol table less than or equal to {@code key}.
       * @param  {any} key the key
       * @return {any}
       */
      floor(key){
        if(_.nichts(key)) throw Error("argument to floor() is null");
        let w,k,it= this.st.keys();
        while(it.hasNext()){
          k=it.next();
          if(k == key || k<key){
            w=k;
          }
        }
        if(w===undefined)
          throw Error("argument to floor() is too small");
        return w;
      }
      static test(){
        let t= new ST();
        t.put("a",1);t.put("g", 9); t.put("c",3); t.put("j",10); t.put("z",26); t.put("x",24);
        console.log(`isEmpty= ${t.isEmpty()}`);
        console.log(`size= ${t.size()}`);
        console.log(`get-c= ${t.get("c")}`);
        console.log(`contains z= ${t.contains("z")}`);
        console.log(`contains m= ${t.contains("m")}`);
        console.log(prnIter(t.keys()));
        console.log(`ceil w= ${t.ceiling("w")}`);
        console.log(`floor k= ${t.floor("k")}`);
        console.log(`min = ${t.min()}`);
        console.log(`max = ${t.max()}`);
        t.remove("x");
        console.log(prnIter(t.keys()));
      }
    }
    //ST.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an ordered symbol table of generic key-value pairs.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class BTree{
      // max children per B-tree node = M-1 (must be even and greater than 2)
      static M = 4;
      Node(k){
        return{
          m:k,    // number of children
          children: new Array(BTree.M) // children
        }
      }
      // internal nodes: only use key and next external nodes: only use key and value
      Entry(key,val,next=UNDEF){ return{ key, val, next } }
      constructor(compareFn){
        //* @property {number} height  height of tree
        //* @property {number} n  number of key-value pairs in the B-tree
        //* @property {object} root root of tree
        this.root = this.Node(0);
        this.compare=compareFn;
        this._height=0;
        this.n=0;
      }
      /**Returns true if this symbol table is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.size() == 0;
      }
      /**Returns the number of key-value pairs in this symbol table.
       * @return {number}
       */
      size(){
        return this.n;
      }
      /**Returns the height of this B-tree (for debugging).
       * @return {number}
       */
      height(){
        return this._height;
      }
      /**Returns the value associated with the given key.
       * @param  key the key
       * @return {any}
       */
      get(key){
        if(_.nichts(key)) throw Error("argument to get() is null");
        return this._search(this.root, key, this._height);
      }
      _search(x, key, ht){
        let cs = x.children;
        // external node
        if(ht == 0){
          for(let j = 0; j < x.m; ++j)
            if(this.compare(key, cs[j].key)==0) return cs[j].val;
        }else{ // internal node
          for(let j = 0; j < x.m; ++j)
            if(j+1 == x.m ||
               this.compare(key, cs[j+1].key)<0)
              return this._search(cs[j].next, key, ht-1);
        }
      }
      /**Inserts the key-value pair into the symbol table, overwriting the old value
       * with the new value if the key is already in the symbol table.
       * If the value is {@code undefined}, this effectively deletes the key from the symbol table.
       * @param  {any} key the key
       * @param  {any} val the value
       */
      put(key, val){
        if(_.nichts(key)) throw Error("argument key to put() is null");
        let t,u = this._insert(this.root, key, val, this._height);
        this.n += 1;
        if(!u) return;
        // need to split root
        t = this.Node(2);
        t.children[0] = this.Entry(this.root.children[0].key, null, this.root);
        t.children[1] = this.Entry(u.children[0].key, null, u);
        this.root = t;
        this._height+=1;
      }
      _insert(h, key, val, ht){
        let j,
            t = this.Entry(key, val);
        if(ht == 0){
          for(j = 0; j < h.m; ++j)
            if(this.compare(key, h.children[j].key)<0) break;
        }else{ // internal node
          for(j = 0; j < h.m; ++j){
            if((j+1 == h.m) ||
               this.compare(key, h.children[j+1].key)<0){
              let u = this._insert(h.children[j++].next, key, val, ht-1);
              if(!u) return null;
              t.key = u.children[0].key;
              t.val = null;
              t.next = u;
              break;
            }
          }
        }
        for(let i = h.m; i > j; --i)
          h.children[i] = h.children[i-1];
        h.children[j] = t;
        h.m++;
        if(h.m >= BTree.M) return this._split(h);
      }
      // split node in half
      _split(h){
        let m2=int(BTree.M/2),
            t = this.Node(m2);
        h.m = m2;
        for(let j = 0; j < m2; ++j)
          t.children[j] = h.children[m2+j];
        return t;
      }
      /**Returns a string representation of this B-tree (for debugging).
       * @return {string}
       */
      toString(){
        function _s(h, ht, indent){
          let s= "", cs= h.children;
          if(ht == 0){
            for(let j = 0; j < h.m; ++j)
              s+= `${indent}${cs[j].key} ${cs[j].val}\n`;
          }else{
            for(let j = 0; j < h.m; ++j){
              if(j > 0)
                s+= `${indent}(${cs[j].key})\n`;
              s+= _s(cs[j].next, ht-1, indent+"     ");
            }
          }
          return s;
        }
        return _s(this.root, this._height, "") + "\n";
      }
      static test(){
        let st = new BTree(CMP);
        st.put("www.cs.princeton.edu", "128.112.136.12");
        st.put("www.cs.princeton.edu", "128.112.136.11");
        st.put("www.princeton.edu",    "128.112.128.15");
        st.put("www.yale.edu",         "130.132.143.21");
        st.put("www.simpsons.com",     "209.052.165.60");
        st.put("www.apple.com",        "17.112.152.32");
        st.put("www.amazon.com",       "207.171.182.16");
        st.put("www.ebay.com",         "66.135.192.87");
        st.put("www.cnn.com",          "64.236.16.20");
        st.put("www.google.com",       "216.239.41.99");
        st.put("www.nytimes.com",      "199.239.136.200");
        st.put("www.microsoft.com",    "207.126.99.140");
        st.put("www.dell.com",         "143.166.224.230");
        st.put("www.slashdot.org",     "66.35.250.151");
        st.put("www.espn.com",         "199.181.135.201");
        st.put("www.weather.com",      "63.111.66.11");
        st.put("www.yahoo.com",        "216.109.118.65");
        console.log("cs.princeton.edu:  " + st.get("www.cs.princeton.edu"));
        console.log("hardvardsucks.com: " + st.get("www.harvardsucks.com"));
        console.log("simpsons.com:      " + st.get("www.simpsons.com"));
        console.log("apple.com:         " + st.get("www.apple.com"));
        console.log("ebay.com:          " + st.get("www.ebay.com"));
        console.log("dell.com:          " + st.get("www.dell.com"));
        console.log("");
        console.log("size:    " + st.size());
        console.log("height:  " + st.height());
        console.log(st.toString());
        console.log("");
      }
    }
    //BTree.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a <em>d</em>-dimensional mathematical vector.
     *  Vectors are mutable: their values can be changed after they are created.
     *  It includes methods for addition, subtraction,
     *  dot product, scalar product, unit vector, and Euclidean norm.
     * @memberof module:mcfud/algo_basic
     * @class
     */
    class SparseVector{
      /**Initializes a d-dimensional zero vector.
       * @param {number} d the dimension of the vector
       */
      constructor(d){
        //* @property {number} d  dimension
        //* @property {ST} st the vector, represented by index-value pairs
        this.d  = d;
        this.st = new ST();
      }
     /**Sets the ith coordinate of this vector to the specified value.
       * @param  {number} i the index
       * @param  {number} value the new value
       */
      put(i, value){
        if(i < 0 || i >= this.d) throw Error("Illegal index");
        //by definition, only store nonzero data
        if(_.feq0(value)) this.st.remove(i);
        else this.st.put(i, value);
      }
      /**Returns the ith coordinate of this vector.
       * @param  {number} i the index
       * @return {number} the value of the ith coordinate of this vector
       */
      get(i){
        if(i < 0 || i >= this.d) throw Error("Illegal index");
        return this.st.contains(i)? this.st.get(i): 0;
      }
      /**Returns the number of nonzero entries in this vector.
       * @return {number}
       */
      nnz(){
        return this.st.size();
      }
      /**Returns the dimension of this vector.
       * @return {number}
       */
      dimension(){
        return this.d;
      }
      /**Returns the inner product of this vector with the specified vector.
       * @param  {SparseVector} that the other vector
       * @return {number} the dot product between this vector and that vector
       */
      dot(that){
        _.assert(that instanceof SparseVector, "expected SparseVector data");
        if(this.d != that.d) throw Error("Vector lengths disagree");
        // iterate over the vector with the fewest nonzeros
        let i,sum=0,
            it= (this.st.size() <= that.st.size()? this : that).st.keys();
        while(it.hasNext()){
          i=it.next();
          if(that.st.contains(i)) sum += this.get(i) * that.get(i);
        }
        return sum;
      }
      /**Returns the inner product of this vector with the specified array.
       * @param  {array} vec the array
       * @return {number} the dot product between this vector and that array
       */
      dotWith(vec){
        let sum = 0;
        for(let i,it= this.st.keys();it.hasNext();){
          i=it.next();
          sum += vec[i] * this.get(i);
        }
        return sum;
      }
      /**Returns the magnitude of this vector.
       * This is also known as the L2 norm or the Euclidean norm.
       * @return {number} the magnitude of this vector
       */
      magnitude(){
        return Math.sqrt(this.dot(this));
      }
      /**Returns the scalar-vector product of this vector with the specified scalar.
       * @param  {number} alpha the scalar
       * @return {number} the scalar-vector product of this vector with the specified scalar
       */
      scale(alpha){
        let c = new SparseVector(this.d);
        for(let i,it= this.st.keys();it.hasNext();){
          i=it.next();
          c.put(i, alpha * this.get(i));
        }
        return c;
      }
      /**Returns the sum of this vector and the specified vector.
       * @param  {SparseVector} that the vector to add to this vector
       * @return {number} the sum of this vector and that vector
       */
      plus(that){
        _.assert(that instanceof SparseVector, "expected SparseVector data");
        if(this.d != that.d) throw Error("Vector lengths disagree");
        let c = new SparseVector(this.d);
        for(let i, it=this.st.keys();it.hasNext();){
          i=it.next();
          c.put(i, this.get(i)); // c = this
        }
        for(let i,it= that.st.keys();it.hasNext();){
          i=it.next();
          c.put(i, that.get(i) + c.get(i));     // c = c + that
        }
        return c;
      }
      /**Returns a string representation of this vector.
       * @return {string}
       */
      toString(){
        let s="";
        for(let i,it= this.st.keys();it.hasNext();){
          i=it.next();
          s+= `(${i}, ${this.st.get(i)}) `;
        }
        return s;
      }
      static test(){
        let a = new SparseVector(10),
            b = new SparseVector(10);
        a.put(3, 0.50);
        a.put(9, 0.75);
        a.put(6, 0.11);
        a.put(6, 0.00);
        b.put(3, 0.60);
        b.put(4, 0.90);
        console.log("a = " + a.toString());
        console.log("b = " + b.toString());
        console.log("a dot b = " + a.dot(b));
        console.log("a + b   = " + a.plus(b).toString());
      }
    }
    //SparseVector.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      prnIter,
      StdCompare:CMP,
      BTree,Bag,Stack,Queue,ST,
      TreeMap,SparseVector,Iterator
    };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //export
    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("mcfud"))
  }else{
    gscope["io/czlab/mcfud/algo/basic"]=_module
  }

})(this);


