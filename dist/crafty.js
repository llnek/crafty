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
  function _module(Mcfud){

    const Basic= Mcfud ? Mcfud["Basic"] : gscope["io/czlab/mcfud/algo/basic"]();
    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();

    const {prnIter,Bag,Stack,Iterator,StdCompare:CMP}= Basic;
    const int=Math.floor;
    const {is,u:_}= Core;

    /**
     * @module mcfud/algo_sort
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // resize the underlying array to have the given capacity
    function resize(c,n,lo,hi,a){
      _.assert(c>n,"bad resize capacity");
      let i, temp = new Array(c);
      for(i=lo; i<hi; ++i) temp[i] = a[i];
      return temp;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const less=(v, w, cmp)=> cmp(v,w) < 0;

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function exch(a, i, j){
      const swap = a[i];
      a[i] = a[j];
      a[j] = swap;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const isSorted=(a,C)=> isSorted3(a, 0, a.length,C);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function isSorted3(a, lo, hi,C){
      for(let i = lo+1; i < hi; ++i)
        if(less(a[i], a[i-1], C)) return false;
      return true;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function show(a){
      let i,s="";
      for(i=0; i<a.length; ++i) s += `${a[i]} `;
      console.log(s);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides static methods for sorting an array using insertion sort.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class Insertion{
      /**Rearranges the array in order specified by the compareFn..
       * @param {array} a the array to be sorted
       * @param {function} compareFn
       * @return {array}
       */
      static sort(a,compareFn){
        const n = a.length;
        for(let i=1; i<n; ++i)
          for(let j=i; j>0 && less(a[j], a[j-1],compareFn); --j){
            exch(a, j, j-1);
          }
        return a;
      }
      /**Rearranges the subarray a[lo..hi) according to the compareFn.
       * @param {array} a the array to be sorted
       * @param {number} lo left endpoint (inclusive)
       * @param {number} hi right endpoint (exclusive)
       * @param {function} compareFn
       * @return {array}
       */
      static sortRange(a, lo, hi,compareFn){
        for(let i=lo+1; i<hi; ++i)
          for(let j=i; j>lo && less(a[j], a[j-1],compareFn); --j){
            exch(a, j, j-1);
          }
        return a;
      }
      /**Returns a permutation that gives the elements in the array
       * according to the compareFn.
       * @param a the array
       * @return {array} a permutation {@code p[]} such that {@code a[p[0]]}, {@code a[p[1]]},
       *    ..., {@code a[p[n-1]]} are in ascending order
       */
      static indexSort(a,compareFn){
        //do not change the original array a[]
        const n=a.length,
              ix = _.fill(n,(i)=> i);
        for(let i=1; i<n; ++i)
          for(let j=i; j>0 && less(a[ix[j]], a[ix[j-1]],compareFn); --j){
            exch(ix, j, j-1)
          }
        return ix;
      }
      static test(){
        let obj="SORTEXAMPLE".split("");
        show(Insertion.sort(obj,CMP));
        obj="bed bug dad yes zoo all bad yet".split(" ");
        show(Insertion.sortRange(obj,0,obj.length,CMP));
        obj="SORTEXAMPLE".split("");
        show(Insertion.indexSort(obj,CMP));
      }
    }
    //Insertion.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides a static method for sorting an array using an optimized
     * binary insertion sort with half exchanges.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class BinaryInsertion{
      /**Rearranges the array according to compareFn.
       * @param {array} a the array to be sorted
       * @param {function} compareFn
       * @return {array}
       */
      static sort(a,compareFn){
        let mid,lo,hi,n=a.length;
        for(let v,i=1; i<n; ++i){
          // binary search to determine index j at which to insert a[i]
          lo = 0; hi = i; v = a[i];
          while(lo<hi){
            mid = lo + _.ndiv(hi-lo,2);
            if(less(v, a[mid],compareFn)) hi = mid;
            else lo = mid + 1;
          }
          // insetion sort with "half exchanges"
          // (insert a[i] at index j and shift a[j], ..., a[i-1] to right)
          for(let j=i; j>lo; --j) a[j] = a[j-1];
          a[lo] = v;
        }
        return a;
      }
      static test(){
        let obj="SORTEXAMPLE".split("");
        show(BinaryInsertion.sort(obj,CMP));
        obj="bed bug dad yes zoo all bad yet".split(" ");
        show(BinaryInsertion.sort(obj,CMP));
      }
    }
    //BinaryInsertion.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides static methods for sorting an array using <em>selection sort</em>.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class Selection{
      /**Rearranges the array in ascending order, using a comparator.
       * @param {array} a the array
       * @param {function} compareFn
       */
      static sort(a, compareFn){
        let min,n=a.length;
        for(let i=0; i<n; ++i){
          min = i;
          for(let j=i+1; j<n; ++j)
            if(less(a[j], a[min],compareFn)) min = j;
          exch(a, i, min);
        }
        return a;
      }
      static test(){
        let obj="SORTEXAMPLE".split("");
        show(Selection.sort(obj,CMP));
        obj="bed bug dad yes zoo all bad yet".split(" ");
        show(Selection.sort(obj,CMP));
      }
    }
    //Selection.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides static methods for sorting an array using <em>Shellsort</em>.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class Shell{
      /**Rearranges the array according to compareFn.
       * @param {array} a the array to be sorted
       * @param {function} compareFn
       * @return {array}
       */
      static sort(a, compareFn){
        let n = a.length,
            h=1, n3= _.ndiv(n,3);
        // 3x+1 increment sequence:  1, 4, 13, 40, 121, 364, 1093, ...
        while(h < n3) h = 3*h + 1;
        while(h >= 1){
          // h-sort the array
          for(let i=h; i<n; ++i){
            for(let j=i; j>=h && less(a[j], a[j-h],compareFn); j -= h)
              exch(a, j, j-h);
          }
          h= _.ndiv(h,3);
        }
        return a;
      }
      static test(){
        let obj="SORTEXAMPLE".split("");
        show(Shell.sort(obj,CMP));
        obj="bed bug dad yes zoo all bad yet".split(" ");
        show(Shell.sort(obj,CMP));
      }
    }
    //Shell.test();

    /***************************************************************************
     *  Index mergesort.
     ***************************************************************************/
    // stably merge a[lo .. mid] with a[mid+1 .. hi] using aux[lo .. hi]
    function mergeIndex(a, index, aux, lo, mid, hi,C){
      // copy to aux[]
      for(let k=lo; k<=hi; ++k){ aux[k] = index[k] }
      // merge back to a[]
      let i = lo, j = mid+1;
      for(let k=lo; k<=hi; ++k){
        if(i>mid) index[k] = aux[j++];
        else if(j>hi) index[k] = aux[i++];
        else if(less(a[aux[j]], a[aux[i]],C)) index[k] = aux[j++];
        else index[k] = aux[i++];
      }
      return a;
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // stably merge a[lo .. mid] with a[mid+1 ..hi] using aux[lo .. hi]
    function merge(a, aux, lo, mid, hi,C){
      // precondition: a[lo .. mid] and a[mid+1 .. hi] are sorted subarrays
      //assert isSorted(a, lo, mid);
      //assert isSorted(a, mid+1, hi);
      // copy to aux[]
      for(let k=lo; k<=hi; ++k){ aux[k] = a[k] }
      // merge back to a[]
      let i=lo, j=mid+1;
      for(let k=lo; k<=hi; ++k){
        if(i>mid) a[k] = aux[j++];
        else if(j>hi) a[k] = aux[i++];
        else if(less(aux[j], aux[i],C)) a[k] = aux[j++];
        else a[k] = aux[i++];
      }
      // postcondition: a[lo .. hi] is sorted
      //assert isSorted(a, lo, hi);
      return a;
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // mergesort a[lo..hi] using auxiliary array aux[lo..hi]
    function sortIndex(a, index, aux, lo, hi,C){
      if(hi<=lo){}else{
        let mid = lo + _.ndiv(hi-lo,2);
        sortIndex(a, index, aux, lo, mid,C);
        sortIndex(a, index, aux, mid + 1, hi,C);
        mergeIndex(a, index, aux, lo, mid, hi,C);
      }
      return a;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides static methods for sorting an array
     * using a top-down, recursive version of <em>mergesort</em>.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class Merge{
      /**Rearranges the array according to compareFn.
       * @param {array} a the array to be sorted
       * @param {function} compareFn
       * @return {array}
       */
      static sort(a,compareFn){
        // mergesort a[lo..hi] using auxiliary array aux[lo..hi]
        function _sort(a, aux, lo, hi,C){
          if(hi<=lo){}else{
            let mid = lo + _.ndiv(hi-lo,2);
            _sort(a, aux, lo, mid,C);
            _sort(a, aux, mid + 1, hi,C);
            merge(a, aux, lo, mid, hi,C);
          }
          return a;
        }
        let aux = new Array(a.length);
        _sort(a, aux, 0, a.length-1,compareFn);
        return a;
      }
      /**Returns a permutation that gives the elements
       * in the array according to the compareFn.
       * @param {array} a the array
       * @param {function} C compare function
       * @return a permutation {@code p[]} such that {@code a[p[0]]}, {@code a[p[1]]},
       *    ..., {@code a[p[N-1]]} are in ascending order
       */
      static indexSort(a,C){
        let n=a.length,
            ix = _.fill(n,(i)=> i);
        let aux = new Array(n);
        sortIndex(a, ix, aux, 0, n-1,C);
        return ix;
      }
      static test(){
        let obj="SORTEXAMPLE".split("");
        show(Merge.sort(obj,CMP));
        obj="bed bug dad yes zoo all bad yet".split(" ");
        show(Merge.sort(obj,CMP));
        obj="SORTEXAMPLE".split("");
        show(Merge.indexSort(obj,CMP));
      }
    }
    //Merge.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides static methods for sorting an array using bubble sort.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class Bubble{
      /**Rearranges the array in ascending order, using the natural order.
       * @param {array} a the array to be sorted
       */
      static sort(a,C){
        const n=a.length;
        for(let x, i=0; i<n; ++i){
          x=0;
          for(let j=n-1; j>i; --j){
            if(less(a[j], a[j-1],C)){
              exch(a, j, j-1);
              ++x;
            }
          }
          if(x == 0) break;
        }
        return a;
      }
      static test(){
        let obj="bed bug dad yes zoo all bad yet".split(" ");
        Bubble.sort(obj,CMP);
        show(obj);
      }
    }
    //Bubble.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // partition the subarray a[lo..hi] so that a[lo..j-1] <= a[j] <= a[j+1..hi]
    // and return the index j.
    function partition(a, lo, hi,C){
      let i = lo,
          v = a[lo],
          j = hi + 1;
      while(true){
        // find item on lo to swap
        while(less(a[++i], v,C)){
          if(i==hi) break;
        }
        // find item on hi to swap
        while(less(v, a[--j],C)){
          if(j==lo) break;// redundant since a[lo] acts as sentinel
        }
        // check if pointers cross
        if(i>=j) break;
        exch(a, i, j);
      }
      // put partitioning item v at a[j]
      exch(a, lo, j);
      // now, a[lo .. j-1] <= a[j] <= a[j+1 .. hi]
      return j;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides static methods for sorting an array and
     * selecting the ith smallest element in an array using quicksort.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class Quick{
      /**Rearranges the array according to the compareFn.
       * @param {array} a the array to be sorted
       * @param {function} compareFn
       */
      static sort(a,compareFn){
        // quicksort the subarray from a[lo] to a[hi]
        function _sort(a, lo, hi,C){
          if(hi<=lo){}else{
            let j = partition(a, lo, hi,C);
            _sort(a, lo, j-1,C);
            _sort(a, j+1, hi,C);
          }
          return a;
        }
        //_.shuffle(a);
        _sort(a, 0, a.length - 1,compareFn);
        return a;
      }
      /**Rearranges the array so that {@code a[k]} contains the kth smallest key;
       * {@code a[0]} through {@code a[k-1]} are less than (or equal to) {@code a[k]}; and
       * {@code a[k+1]} through {@code a[n-1]} are greater than (or equal to) {@code a[k]}.
       *
       * @param  {array} a the array
       * @param  {number} k the rank of the key
       * @param {function} compareFn
       * @return the key of rank {@code k}
       */
      static select(a, k,compareFn){
        if(k < 0 || k >= a.length)
          throw Error(`index is not between 0 and ${a.length}: ${k}`);
        //_.shuffle(a);
        let lo = 0, hi = a.length-1;
        while(hi>lo){
          let i= partition(a, lo, hi, compareFn);
          if(i>k) hi = i-1;
          else if(i<k) lo = i+1;
          else return a[i];
        }
        return a[lo];
      }
      static test(){
        let obj="bed bug dad yes zoo all bad yet".split(" ");
        show(Quick.sort(obj,CMP));
        obj="SORTEXAMPLE".split("");
        show(Quick.sort(obj, CMP));
        _.shuffle(obj)
        obj.forEach((s,i)=> console.log(Quick.select(obj,i,CMP)));
      }
    }
    //Quick.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const less4=(a,i, j,C)=> less(a[i], a[j],C);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // is pq[1..n] a max heap?
    function isMaxHeap(M){
      for(let i=1; i <= M.n; ++i)
        if(_.nichts(M.pq[i])) return false;

      for(let i = M.n+1; i < M.pq.length; ++i)
        if(!_.nichts(M.pq[i])) return false;

      if(!_.nichts(M.pq[0])) return false;
      return isMaxHeapOrdered(1,M);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // is subtree of pq[1..n] rooted at k a max heap?
    function isMaxHeapOrdered(k,M){
      if(k > M.n) return true;
      let left = 2*k,
          right = 2*k + 1;
      if(left  <= M.n &&
         less4(M.pq, k, left,M.comparator))  return false;
      if(right <= M.n &&
         less4(M.pq, k, right,M.comparator)) return false;
      return isMaxHeapOrdered(left,M) && isMaxHeapOrdered(right,M);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a priority queue of generic keys.
     *  It supports the usual insert and delete-the-minimum operations,
     *  along with the merging of two heaps together.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class FibonacciMinPQ{
      Node(key){
        //int order; //Order of the tree rooted by this Node
        return {key, order:0};// prev:null, next:null, child:null
      }
      constructor(compareFn, keys){
        //private Node head;          //Head of the circular root list
        //private Node min;         //Minimum Node of the root list
        //private int size;         //Number of keys in the heap
        //private final Comparator<Key> comp; //Comparator over the keys
        //private HashMap<Integer, Node> table = new HashMap<Integer, Node>(); //Used for the consolidate operation
        this.compare=compareFn;
        this.table=new Map();
        this.head=UNDEF;
        this._min=UNDEF;
        this.n=0;
        if(is.vec(keys))
          keys.forEach(k=> this.insert(k));
      }
      /**Whether the priority queue is empty
      * @return {boolean}
      */
      isEmpty(){
        return this.n == 0;
      }
      /**Number of elements currently on the priority queue
      * @return {number}
      */
      size(){
        return this.n;
      }
      /**Insert a key in the queue
      * @param {any} key a Key
      */
      insert(key){
        let x = this.Node(key);
        this.n+= 1;
        this.head = this._insertNode(x, this.head);
        this._min= !this._min? this.head
                           : (this._greater(this._min.key, key) ? this.head : this._min);
      }
      /**Gets the minimum key currently in the queue
      * @return {any}
      */
      min(){
        if(this.isEmpty())
          throw Error("Priority queue is empty");
        return this._min.key;
      }
      /**Deletes the minimum key
      * @return {any} the minimum key
      */
      delMin(){
        if(this.isEmpty())
          throw Error("Priority queue is empty");
        this.head = this._cut(this._min, this.head);
        let x= this._min.child,
            key = this._min.key;
        this._min.key = UNDEF;
        if(x){
          this.head = this._meld(this.head, x);
          this._min.child = UNDEF;
        }
        this.n -= 1;
        if(!this.isEmpty()) this._consolidate();
        else this._min = UNDEF;
        return key;
      }
      /**Merges two heaps together
      * This operation is destructive
      * @param {FibonacciMinPQ} that a Fibonacci heap
      * @return {FibonacciMinPQ}
      */
      union(that){
        this.head = this._meld(this.head, that.head);
        this._min = this._greater(this._min.key, that._min.key) ? that._min : this._min;
        this.n = this.n + that.n;
        return this;
      }
      _greater(n, m){
        if(_.nichts(n)) return false;
        if(_.nichts(m)) return true;
        return this.compare(n,m) > 0;
      }
      //Assuming root1 holds a greater key than root2, root2 becomes the new root
      _link(root1, root2){
        root2.child = this._insertNode(root1, root2.child);
        root2.order+=1;
      }
      //Coalesce the roots, thus reshapes the tree
      _consolidate(){
        this.table.clear();
        let x = this.head,
            y = UNDEF,
            z = UNDEF,
            maxOrder = 0;
        this._min = this.head;
        do{
          y = x;
          x = x.next;
          z = this.table.get(y.order);
          while(z){
            this.table.delete(y.order);
            if(this._greater(y.key, z.key)){
              this._link(y, z);
              y = z;
            }else{
              this._link(z, y);
            }
            z = this.table.get(y.order);
          }
          this.table.set(y.order, y);
          if(y.order > maxOrder) maxOrder = y.order;
        }while(x !== this.head);
        this.head = null;
        this.table.forEach((v)=>{
          if(v){
            this._min = this._greater(this._min.key, v.key) ? v : this._min;
            this.head = this._insertNode(v, this.head);
          }
        })
      }
      //Inserts a Node in a circular list containing head, returns a new head
      _insertNode(x, head){
        if(!head){
          x.prev = x;
          x.next = x;
        }else{
          head.prev.next = x;
          x.next = head;
          x.prev = head.prev;
          head.prev = x;
        }
        return x;
      }
      //Removes a tree from the list defined by the head pointer
      _cut(x, head){
        if(x.next === x) {
          x.next = UNDEF;
          x.prev = UNDEF;
          return UNDEF;
        }else{
          x.next.prev = x.prev;
          x.prev.next = x.next;
          let res = x.next;
          x.next = UNDEF;
          x.prev = UNDEF;
          return head === x?  res: head;
        }
      }
      //Merges two root lists together
      _meld(x, y){
        if(!x) return y;
        if(!y) return x;
        x.prev.next = y.next;
        y.next.prev = x.prev;
        x.prev = y;
        y.next = x;
        return x;
      }
      /**Gets an Iterator over the Keys in the priority queue in ascending order
      * The Iterator does not implement the remove() method
      * iterator() : Worst case is O(n)
      * next() :  Worst case is O(log(n)) (amortized)
      * hasNext() :   Worst case is O(1)
      * @return {Iterator}
      */
      iter(){
        let copy = new FibonacciMinPQ(this.compare);
        let insertAll=(head)=>{
          if(!head) return;
          let x = head;
          do{
            copy.insert(x.key);
            insertAll(x.child);
            x = x.next;
          }while (x !== head);
        };
        insertAll(this.head);
        return{
          remove(){ throw Error("UnsupportedOperationException") },
          hasNext(){ return !copy.isEmpty() },
          next(){
            if(!this.hasNext()) throw Error("NoSuchElementException");
            return copy.delMin();
          }
        }
      }
      static test(){
        let msg="",
            obj= new FibonacciMinPQ(CMP);
        "PQE".split("").forEach(s=>obj.insert(s));
        msg += obj.delMin() + " ";
        "XAM".split("").forEach(s=>obj.insert(s));
        msg += obj.delMin() + " ";
        "PLE".split("").forEach(s=>obj.insert(s));
        msg += obj.delMin() + " ";
        obj.isEmpty();
        console.log(msg)
        console.log("min= " + obj.min());
        console.log(prnIter(obj.iter()));
        console.log("(" + obj.size() + " left on pq)");
        let obj2 = new FibonacciMinPQ(CMP);
        "ZTAK".split("").forEach(s=> obj2.insert(s));
        obj2= obj2.union(obj);
        console.log(prnIter(obj2.iter()));
      }
    }
    //FibonacciMinPQ.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an indexed priority queue of generic keys.
     *  It supports the usual insert and delete-the-minimum operations,
     *  along with delete and change-the-key methods.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class IndexFibonacciMinPQ{
      Node(key){
        //Node<Key> prev, next;     //siblings of the Node
        ////Node<Key> parent, child;    //parent and child of this Node
        //boolean mark;         //Indicates if this Node already lost a child
        return{key, order:0, index:0};//
        //prev:null, next:null, parent:null, child:null, mark:false
      }
      constructor(maxN,compareFn){
        //private Node<Key>[] nodes;      //Array of Nodes in the heap
        //private Node<Key> head;       //Head of the circular root list
        //private Node<Key> min;        //Minimum Node in the heap
        //private int size;         //Number of keys in the heap
        //private int n;            //Maximum number of elements in the heap
        //private HashMap<Integer, Node<Key>> table = new HashMap<Integer, Node<Key>>(); //Used for the consolidate operation
        if(maxN < 0)
          throw Error("Cannot create a priority queue of negative size");
        this.maxN = maxN;
        this.n=0;
        this.head=UNDEF;
        this._min=UNDEF;
        this.compare = compareFn;
        this.table=new Map();
        this.nodes = new Array(maxN);
      }
      /**Whether the priority queue is empty
      * @return {boolean}
      */
      isEmpty(){
        return this.n== 0;
      }
      /**Does the priority queue contains the index i ?
      * @param {number} i an index
      * @return {boolean}
      */
      contains(i){
        if(i<0 || i >= this.maxN) throw Error("IllegalArgumentException");
        return _.echt(this.nodes[i]);
      }
      /**Number of elements currently on the priority queue
      * @return {number}
      */
      size(){
        return this.n;
      }
      /**Associates a key with an index
      * @param {number} i an index
      * @param {any} key a Key associated with i
      */
      insert(i, key){
        if(i<0 || i>= this.maxN) throw Error("IllegalArgumentException");
        if(this.contains(i)) throw Error("Specified index is already in the queue");
        let x = this.Node(key);
        x.index = i;
        this.nodes[i] = x;
        this.n+=1;
        this.head = this._insertNode(x, this.head);
        this._min= !this._min? this.head
                             : this._greater(this._min.key, key) ? this.head : this._min;
      }
      /**Get the index associated with the minimum key
      * @return {number} the index associated with the minimum key
      */
      minIndex(){
        if(this.isEmpty()) throw Error("Priority queue is empty");
        return this._min.index;
      }
      /**Get the minimum key currently in the queue
      * @return {any} the minimum key currently in the priority queue
      */
      min(){
        if(this.isEmpty()) throw Error("Priority queue is empty");
        return this._min.key;
      }
      /**Delete the minimum key
      * @return {number} the index associated with the minimum key
      */
      delMin(){
        if(this.isEmpty()) throw Error("Priority queue is empty");
        this.head = this._cutNode(this._min, this.head);
        let x = this._min.child,
            index = this._min.index;
        this._min.key = UNDEF;
        if(x){
          do{
            x.parent = UNDEF;
            x = x.next;
          }while(x !== this._min.child);
          this.head = this._meld(this.head, x);
          this._min.child = UNDEF;     //For garbage collection
        }
        this.n-=1;
        if(!this.isEmpty()) this._consolidate();
        else this._min = UNDEF;
        this.nodes[index] = UNDEF;
        return index;
      }
      /**Get the key associated with index i
      * @param {number} i an index
      * @return {any} the key associated with index i
      */
      keyOf(i){
        if(i< 0 || i >= this.maxN) throw Error("IllegalArgumentException");
        if(!this.contains(i)) throw Error("Specified index is not in the queue");
        return this.nodes[i].key;
      }
      /**Changes the key associated with index i to the given key
      * If the given key is greater, Worst case is O(log(n))
      * If the given key is lower, Worst case is O(1) (amortized)
      * @param {number} i an index
      * @param {any} key the key to associate with i
      */
      changeKey(i, key){
        if(i < 0 || i >= this.maxN) throw Error("IllegalArgumentException");
        if(!this.contains(i)) throw Error("Specified index is not in the queue");
        this._greater(key, this.nodes[i].key)? this.increaseKey(i, key) : this.decreaseKey(i, key);
      }
      /**Decreases the key associated with index i to the given key
      * @param {number} i an index
      * @param {any} key the key to associate with i
      */
      decreaseKey(i, key){
        if(i<0 || i >= this.maxN) throw Error("IllegalArgumentException");
        if(!this.contains(i)) throw Error("Specified index is not in the queue");
        if(this._greater(key, this.nodes[i].key))
          throw Error("Calling with this argument would not decrease the key");
        let x = this.nodes[i];
        x.key = key;
        if(this._greater(this._min.key, key)){
          this._min = x;
        }
        if(x.parent && this._greater(x.parent.key, key)){
          this._cut(i)
        }
      }
      /**Increases the key associated with index i to the given key
      * @param {number} i an index
      * @param {any} key the key to associate with i
      */
      increaseKey(i, key){
        if(i<0 || i>= this.maxN) throw Error("IllegalArgumentException");
        if(!this.contains(i)) throw Error("Specified index is not in the queue");
        if(this._greater(this.nodes[i].key, key))
          throw Error("Calling with this argument would not increase the key");
        this.delete(i);
        this.insert(i, key);
      }
      /**Deletes the key associated the given index
      * @param {number} i an index
      */
      delete(i){
        if(i<0 || i >= this.maxN) throw Error("IllegalArgumentException");
        if(!this.contains(i)) throw Error("Specified index is not in the queue");
        let x = this.nodes[i];
        x.key = null;       //For garbage collection
        if(x.parent){ this._cut(i) }
        this.head = this._cutNode(x, this.head);
        if(x.child){
          let child = x.child;
          x.child = UNDEF;     //For garbage collection
          x = child;
          do{
            child.parent = UNDEF;
            child = child.next;
          }while(child !== x);
          this.head = this._meld(this.head, child);
        }
        if(!this.isEmpty()) this._consolidate();
        else this._min = UNDEF;
        this.nodes[i] = UNDEF;
        this.n-=1;
      }
      _greater(n, m){
        if(_.nichts(n)) return false;
        if(_.nichts(m)) return true;
        return this.compare(n, m) > 0;
      }
      _link(root1, root2){
        root1.parent = root2;
        root2.child = this._insertNode(root1, root2.child);
        root2.order+=1;
      }
      //Removes a Node from its parent's child list and insert it in the root list
      //If the parent Node already lost a child, reshapes the heap accordingly
      _cut(i){
        let x = this.nodes[i];
        let parent = x.parent;
        parent.child = this._cutNode(x, parent.child);
        x.parent = UNDEF;
        parent.order-=1;
        this.head = this._insertNode(x, this.head);
        parent.mark = !parent.mark;
        if(!parent.mark && parent.parent){
          this._cut(parent.index);
        }
      }
      //Coalesces the roots, thus reshapes the heap
      //Caching a HashMap improves greatly performances
      _consolidate(){
        let y = UNDEF,
            z = UNDEF,
            maxOrder = 0,
            x = this.head;
        this.table.clear();
        this._min = this.head;
        do{
          y = x;
          x = x.next;
          z = this.table.get(y.order);
          while(z){
            this.table.delete(y.order);
            if(this._greater(y.key, z.key)){
              this._link(y, z);
              y = z;
            }else{
              this._link(z, y);
            }
            z = this.table.get(y.order);
          }
          this.table.set(y.order, y);
          if(y.order > maxOrder) maxOrder = y.order;
        }while(x !== this.head);
        this.head = UNDEF;
        this.table.forEach(n=>{
          this._min = this._greater(this._min.key, n.key) ? n : this._min;
          this.head = this._insertNode(n, this.head);
        })
      }
      //Inserts a Node in a circular list containing head, returns a new head
      _insertNode(x, head){
        if(!head){
          x.prev = x;
          x.next = x;
        }else{
          head.prev.next = x;
          x.next = head;
          x.prev = head.prev;
          head.prev = x;
        }
        return x;
      }
      //Removes a tree from the list defined by the head pointer
      _cutNode(x, head){
        if(x.next === x){
          x.next = UNDEF;
          x.prev = UNDEF;
          return UNDEF;
        }else{
          x.next.prev = x.prev;
          x.prev.next = x.next;
          let res = x.next;
          x.next = UNDEF;
          x.prev = UNDEF;
          return head === x?  res: head;
        }
      }
      _meld(x, y){
        if(!x) return y;
        if(!y) return x;
        x.prev.next = y.next;
        y.next.prev = x.prev;
        x.prev = y;
        y.next = x;
        return x;
      }
      /**Get an Iterator over the indexes in the priority queue in ascending order
      * The Iterator does not implement the remove() method
      * iterator() : Worst case is O(n)
      * next() :  Worst case is O(log(n)) (amortized)
      * hasNext() :   Worst case is O(1)
      * @return {Iterator}
      */
      iter(){
        let copy= new IndexFibonacciMinPQ(this.maxN,this.compare);
        this.nodes.forEach(x=> {
          if(x) copy.insert(x.index, x.key);
        });
        return{
          remove(){ throw Error("UnsupportedOperationException") },
          hasNext(){ return !copy.isEmpty() },
          next(){
            if(!this.hasNext()) throw Error("NoSuchElementException");
            return copy.delMin();
          }
        }
      }
      static test(){
        let strings = [ "it", "was", "the", "best", "of", "times", "it", "was", "the", "worst" ];
        let pq = new IndexFibonacciMinPQ(strings.length,CMP);
        for(let i=0; i<strings.length; ++i) pq.insert(i, strings[i]);
        // delete and print each key
        console.log("min= " +pq.min());
        console.log("minindex= "+pq.minIndex());
        console.log("size= "+pq.size());
        console.log("contains(3)="+pq.contains(3));
        console.log("keyOf(3)="+pq.keyOf(3));
        pq.changeKey(3,"bbbb");
        //pq.delete(3);
        while(!pq.isEmpty()){
          let i = pq.minIndex();
          console.log(i + " " + pq.keyOf(i));
          pq.delMin();
        }
        console.log("");
        // reinsert the same strings
        for(let i=0; i<strings.length; ++i) pq.insert(i, strings[i]);
        // print each key using the iterator
        for(let i,it=pq.iter();it.hasNext();){
          i=it.next();
          console.log(i + " " + strings[i]);
        }
        while(!pq.isEmpty()){ pq.delMin() }
      }
    }
    //IndexFibonacciMinPQ.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a priority queue of generic keys.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class MinPQ{
      /**Initializes an empty priority queue with the given initial capacity.
       * @param {function} compareFn
       * @param {number|array} keys capacity or keys
       */
      constructor(compareFn, keys){
        //* @property {function} comparator
        //* @property {number} n // number of items on priority queue
        //* @property {array} pq // store items at indices 1 to n
        this.comparator = compareFn;
        this.n=0;
        if(is.vec(keys)){
          this.pq = new Array(keys.length+1);
          this.n = keys.length;
          for(let i=0; i< this.n; ++i) this.pq[i+1] = keys[i];
          for(let k = int(this.n/2); k>=1; --k) this._sink(k,this);
        }else{
          this.pq= new Array(is.num(keys)? keys: 2);
        }
        _.assert(this._isMinHeap(),"not min heap");
      }
      /**Returns true if this priority queue is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.n == 0;
      }
      /**Returns the number of keys on this priority queue.
       * @return {number}
       */
      size(){
        return this.n;
      }
      /**Returns a smallest key on this priority queue.
       * @return {any}
       */
      min(){
        if(this.isEmpty()) throw Error("Priority queue underflow");
        return this.pq[1];
      }
      /**Adds a new key to this priority queue.
       * @param  x the key to add to this priority queue
       */
      insert(x){
        // double size of array if necessary
        if(this.n==this.pq.length-1)
          this.pq=resize(2*this.pq.length, this.n, 1, this.n+1, this.pq);
        // add x, and percolate it up to maintain heap invariant
        this.pq[++this.n] = x;
        this._swim(this.n);
        _.assert(this._isMinHeap(),"not min heap-insert");
      }
      /**Removes and returns a smallest key on this priority queue.
       * @return {any}
       */
      delMin(){
        if(this.isEmpty()) throw Error("Priority queue underflow");
        let min=this.pq[1];
        exch(this.pq, 1, this.n--);
        this._sink(1);
        this.pq[this.n+1] = UNDEF;// to avoid loitering and help with garbage collection
        if((this.n>0) &&
           (this.n== _.ndiv(this.pq.length-1,4)))
          this.pq= resize(_.ndiv(this.pq.length,2),this.n,1,this.n+1,this.pq);
        return min;
      }
      _swim(k){
        while(k>1 && this._greater(_.ndiv(k,2), k)){
          exch(this.pq, k, _.ndiv(k,2));
          k=_.ndiv(k,2);
        }
      }
      _sink(k){
        while(2*k <= this.n){
          let j = 2*k;
          if(j<this.n && this._greater(j, j+1)) j++;
          if(!this._greater(k, j)) break;
          exch(this.pq, k, j);
          k=j;
        }
      }
      _greater(i, j){
        return this.comparator(this.pq[i], this.pq[j]) > 0;
      }
      // is pq[1..n] a min heap?
      _isMinHeap(){
        for(let i=1; i<=this.n; ++i) if(_.nichts(this.pq[i])) return false;
        for(let i=this.n+1; i<this.pq.length; ++i) if(!_.nichts(this.pq[i])) return false;
        return _.echt(this.pq[0])? false: this._isMinHeapOrdered(1);
      }
      // is subtree of pq[1..n] rooted at k a min heap?
      _isMinHeapOrdered(k){
        if(k>this.n) return true;
        let left = 2*k,
            right = 2*k + 1;
        if(left  <= this.n && this._greater(k, left))  return false;
        if(right <= this.n && this._greater(k, right)) return false;
        return this._isMinHeapOrdered(left) && this._isMinHeapOrdered(right);
      }
      /**Returns an iterator that iterates over the keys.
       * @return {Iterator}
       */
      iter(){
        // add all items to copy of heap takes
        // linear time since already in heap order so no keys move
        let copy = new MinPQ(this.comparator, this.size());
        for(let i=1; i<=this.n; ++i) copy.insert(this.pq[i]);
        return{
          remove(){ throw Error("UnsupportedOperationException") },
          hasNext(){ return !copy.isEmpty() },
          next(){
            if(!this.hasNext()) throw Error("NoSuchElementException");
            return copy.delMin();
          }
        }
      }
      static test(){
        let msg="",
            obj= new MinPQ(CMP);
        "PQE".split("").forEach(s=>obj.insert(s));
        msg += obj.delMin() + " ";
        "XAM".split("").forEach(s=>obj.insert(s));
        msg += obj.delMin() + " ";
        "PLE".split("").forEach(s=>obj.insert(s));
        msg += obj.delMin() + " ";
        console.log(msg)
        console.log("(" + obj.size() + " left on pq)");
      }
    }
    //MinPQ.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a priority queue of generic keys.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class MaxPQ{
      /**Initializes an empty priority queue with the given initial capacity,
       * using the given comparator.
       * @param {function} compareFn
       * @param {any} keys
       */
      constructor(compareFn, keys){
        //* @property {function} comparator
        //* @property {number} n // number of items on priority queue
        //* @property {array} pq // store items at indices 1 to n
        this.comparator = compareFn;
        this.n=0;
        if(is.vec(keys)){
          this.pq = new Array(keys.length+1);
          this.n = keys.length;
          for(let i=0; i<this.n; ++i) this.pq[i+1] = keys[i];
          for(let k=int(this.n/2); k>=1; --k) this._sink(k);
        }else{
          this.pq= new Array(is.num(keys)? keys: 2);
        }
        _.assert(this._isMaxHeap(),"not max heap");
      }
      /**Returns true if this priority queue is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.n == 0
      }
      /**Returns the number of keys on this priority queue.
       * @return {number}
       */
      size(){
        return this.n
      }
      /**Returns a largest key on this priority queue.
       * @return {any}
       */
      max(){
        if(this.isEmpty())
          throw Error("Priority queue underflow");
        return this.pq[1];
      }
      /**Adds a new key to this priority queue.
       * @param  {any} x
       */
      insert(x){
        // double size of array if necessary
        if(this.n == this.pq.length-1)
          this.pq=resize(2*this.pq.length, this.n, 1,this.n+1, this.pq);
        // add x, and percolate it up to maintain heap invariant
        this.n+=1;
        this.pq[this.n] = x;
        this._swim(this.n);
        _.assert(this._isMaxHeap(),"not max heap-insert");
      }
      /**Removes and returns a largest key on this priority queue.
       * @return a largest key on this priority queue
       * @throws Error if this priority queue is empty
       */
      delMax(){
        if(this.isEmpty())
          throw Error("Priority queue underflow");
        let max = this.pq[1];
        exch(this.pq, 1, this.n);
        this.n-=1;
        this._sink(1);
        this.pq[this.n+1] = null;     // to avoid loitering and help with garbage collection
        if(this.n > 0 &&
           this.n == _.ndiv(this.pq.length-1,4))
          this.pq=resize(_.ndiv(this.pq.length,2), this.n, 1, this.n+1, this.pq);
        return max;
      }
      _isMaxHeap(){
        for(let i=1; i <= this.n; ++i) if(_.nichts(this.pq[i])) return false;
        for(let i = this.n+1; i < this.pq.length; ++i) if(_.echt(this.pq[i])) return false;
        if(_.echt(this.pq[0])) return false;
        return this._isMaxHeapOrdered(1);
      }
      _isMaxHeapOrdered(k){
        if(k > this.n) return true;
        let left = 2*k,
            right = 2*k + 1;
        if(left  <= this.n && less4(this.pq,k, left,this.comparator))  return false;
        if(right <= this.n && less4(this.pq,k, right,this.comparator)) return false;
        return this._isMaxHeapOrdered(left) && this._isMaxHeapOrdered(right);
      }
      _swim(k){
        while(k>1 && less4(this.pq, _.ndiv(k,2), k, this.comparator)){
          exch(this.pq, k, _.ndiv(k,2));
          k= _.ndiv(k,2);
        }
      }
      _sink(k){
        let j;
        while(2*k <= this.n){
          j = 2*k;
          if(j<this.n && less4(this.pq, j, j+1,this.comparator)) ++j;
          if(!less4(this.pq, k, j, this.comparator)) break;
          exch(this.pq, k, j);
          k=j;
        }
      }
      /**Returns an iterator that iterates over the keys.
       * @return {Iterator}
       */
      iter(){
        // add all items to copy of heap takes linear time since already in heap order so no keys move
        const copy = new MaxPQ(this.comparator, this.size());
        for(let i=1; i<=this.n; ++i) copy.insert(this.pq[i]);
        return{
          remove(){ throw Error("UnsupportedOperationException") },
          hasNext(){ return !copy.isEmpty() },
          next(){
            if(!this.hasNext()) throw Error("NoSuchElementException");
            return copy.delMax();
          }
        }
      }
      static test(){
        let msg="",
            obj= new MaxPQ(CMP);
        "PQE".split("").forEach(s=>obj.insert(s));
        msg += obj.delMax() + " ";
        "XAM".split("").forEach(s=>obj.insert(s));
        msg += obj.delMax() + " ";
        "PLE".split("").forEach(s=>obj.insert(s));
        msg += obj.delMax() + " ";
        console.log(msg)
        console.log("(" + obj.size() + " left on pq)");
      }
    }
    //MaxPQ.test();

    /***************************************************************************
     * Helper functions for comparisons and swaps.
     * Indices are "off-by-one" to support 1-based indexing.
     ***************************************************************************/
    function lessOneOff(pq, i, j, C){
      return C(pq[i-1], pq[j-1]) < 0
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function exchOneOff(pq, i, j){
      const swap = pq[i-1];
      pq[i-1] = pq[j-1];
      pq[j-1] = swap;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Provides a static method to sort an array using <em>heapsort</em>.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class Heap{
      /**Rearranges the array according to the compareFn.
       * @param {array} pq the array to be sorted
       * @param {function} compareFn
       */
      static sort(pq,compareFn){
        function _sink4(pq, k, n,C){
          while(2*k <= n){
            let j = 2*k;
            if(j<n && lessOneOff(pq, j, j+1,C)) ++j;
            if(!lessOneOff(pq, k, j,C)) break;
            exchOneOff(pq, k, j);
            k=j;
          }
        }
        let k,n=pq.length;
        // heapify phase
        for(k = _.ndiv(n,2); k >= 1; --k){
          _sink4(pq, k, n, compareFn)
        }
        // sortdown phase
        k=n;
        while(k > 1){
          exchOneOff(pq, 1, k--);
          _sink4(pq, 1, k,compareFn);
        }
        //////
        return pq;
      }
      static test(){
        let obj="SORTEXAMPLE".split("");
        show(Heap.sort(obj,CMP));
        obj="bed bug dad yes zoo all bad yet".split(" ");
        show(Heap.sort(obj,CMP));
      }
    }
    //Heap.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an indexed priority queue of generic keys.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class IndexMinPQ{
      /**
       * Initializes an empty indexed priority queue with indices between {@code 0}
       * and {@code maxN - 1}.
       * @param {number} maxN the keys on this queue are index from {@code 0} {@code maxN - 1}
       * @param {function} compareFn
       */
      constructor(maxN,compareFn){
        //* @property {number} maxN  maximum number of elements on PQ
        //* @property {number} n number of elements on PQ
        //* @property {array} pq  binary heap using 1-based indexing
        //* @property {array} qp  inverse of pq - qp[pq[i]] = pq[qp[i]] = i
        //* @property {array} mKeys  keys[i] = priority of i
        if(maxN < 0) throw Error(`IllegalArgumentException`);
        this.compare=compareFn;
        this.maxN = maxN;
        this.n = 0;
        this.mKeys = new Array(maxN+1);// make this of length maxN??
        this.pq = new Array(maxN + 1);
        this.qp = new Array(maxN + 1); // make this of length maxN??
        for(let i=0; i<=maxN; ++i) this.qp[i] = -1;
      }
      /**Returns true if this priority queue is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.n == 0;
      }
      /**Is {@code i} an index on this priority queue?
       * @param  {number} i an index
       * @return {boolean}
       */
      contains(i){
        this._validateIndex(i);
        return this.qp[i] != -1;
      }
      /**Returns the number of keys on this priority queue.
       * @return {number}
       */
      size(){
        return this.n;
      }
      /**Associates key with index {@code i}.
       * @param  {number} i an index
       * @param  {any} key the key to associate with index {@code i}
       */
      insert(i, key){
        this._validateIndex(i);
        if(this.contains(i))
          throw Error("index is already in the priority queue");
        ++this.n;
        this.qp[i] = this.n;
        this.pq[this.n] = i;
        this.mKeys[i] = key;
        this._swim(this.n);
      }
      /**Returns an index associated with a minimum key.
       * @return {any}
       */
      minIndex(){
        if(this.n == 0) throw Error("Priority queue underflow");
        return this.pq[1];
      }
      /**Returns a minimum key.
       * @return {any}
       */
      minKey(){
        if(this.n == 0) throw Error("Priority queue underflow");
        return this.mKeys[this.pq[1]];
      }
      /**Removes a minimum key and returns its associated index.
       * @return {any}
       */
      delMin(){
        if(this.n == 0) throw Error("Priority queue underflow");
        let min = this.pq[1];
        this._exch(1, this.n--);
        this._sink(1);
        _.assert(min == this.pq[this.n+1], "No good");
        this.qp[min] = -1; // delete
        this.mKeys[min] = null;  // to help with garbage collection
        this.pq[this.n+1] = -1; // not needed
        return min;
      }
      /**Returns the key associated with index {@code i}.
       * @param  {number} i the index of the key to return
       * @return {any}
       */
      keyOf(i){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        return this.mKeys[i];
      }
      /**Change the key associated with index {@code i} to the specified value.
       * @param  {number} i the index of the key to change
       * @param  {any} key change the key associated with index {@code i} to this key
       */
      changeKey(i, key){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        this.mKeys[i] = key;
        this._swim(this.qp[i]);
        this._sink(this.qp[i]);
      }
      /**Decrease the key associated with index {@code i} to the specified value.
       * @param  {number} i the index of the key to decrease
       * @param  {any} key decrease the key associated with index {@code i} to this key
       */
      decreaseKey(i, key){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        let c=this.compare(this.mKeys[i],key);
        if(c== 0)
          throw Error("Calling decreaseKey() with a key equal to the key in the priority queue");
        if(c< 0)
          throw Error("Calling decreaseKey() with a key strictly greater than the key in the priority queue");
        this.mKeys[i] = key;
        this._swim(this.qp[i]);
      }
      /**Increase the key associated with index {@code i} to the specified value.
       * @param  {number} i the index of the key to increase
       * @param  {any} key increase the key associated with index {@code i} to this key
       */
      increaseKey(i, key){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        let c= this.compare(this.mKeys[i],key);
        if(c==0)
          throw Error("Calling increaseKey() with a key equal to the key in the priority queue");
        if(c>0)
          throw Error("Calling increaseKey() with a key strictly less than the key in the priority queue");
        this.mKeys[i] = key;
        this._sink(this.qp[i]);
      }
      /**Remove the key associated with index {@code i}.
       * @param  {number} i the index of the key to remove
       */
      delete(i){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        let index = this.qp[i];
        this._exch(index, this.n--);
        this._swim(index);
        this._sink(index);
        this.mKeys[i] = UNDEF;
        this.qp[i] = -1;
      }
      _validateIndex(i){
        if(i<0) throw Error("index is negative: " + i);
        if(i >= this.maxN) throw Error("index >= capacity: " + i);
      }
      _greater(i, j){
        return this.compare(this.mKeys[this.pq[i]],this.mKeys[this.pq[j]]) > 0;
      }
      _exch(i, j){
        let swap = this.pq[i];
        this.pq[i] = this.pq[j];
        this.pq[j] = swap;
        this.qp[this.pq[i]] = i;
        this.qp[this.pq[j]] = j;
      }
      _swim(k){
        while(k>1 && this._greater(_.ndiv(k,2), k)){
          this._exch(k, _.ndiv(k,2));
          k = _.ndiv(k,2);
        }
      }
      _sink(k){
        while(2*k <= this.n){
          let j = 2*k;
          if(j<this.n && this._greater(j, j+1)) ++j;
          if(!this._greater(k, j)) break;
          this._exch(k, j);
          k = j;
        }
      }
      /**Returns an iterator that iterates over the keys on the
       * priority queue in ascending order.
       * @return {Iterator}
       */
      iter(){
        // create a new pq
        let copy= new IndexMinPQ(this.pq.length-1, this.compare);
        // add all elements to copy of heap
        // takes linear time since already in heap order so no keys move
        for(let i=1; i <= this.n; ++i)
          copy.insert(this.pq[i], this.mKeys[this.pq[i]]);
        return{
          remove(){ throw Error(`UnsupportedOperationException`) },
          hasNext(){ return !copy.isEmpty() },
          next(){
            if(!this.hasNext()) throw Error(`NoSuchElementException`);
            return copy.delMin();
          }
        }
      }
      static test(){
        // insert a bunch of strings
        let strings = [ "it", "was", "the", "best", "of", "times", "it", "was", "the", "worst" ];
        let pq = new IndexMinPQ(strings.length,CMP);
        for(let i=0; i<strings.length; ++i) pq.insert(i, strings[i]);
        // delete and print each key
        while(!pq.isEmpty()){
          let i = pq.delMin();
          console.log(i + " " + strings[i]);
        }
        console.log("");
        // reinsert the same strings
        for(let i=0; i<strings.length; ++i) pq.insert(i, strings[i]);
        // print each key using the iterator
        for(let i,it=pq.iter();it.hasNext();){
          i=it.next();
          console.log(i + " " + strings[i]);
        }
        while(!pq.isEmpty()){ pq.delMin() }
      }
    }
    //IndexMinPQ.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an indexed priority queue of generic keys.
     * @memberof module:mcfud/algo_sort
     * @class
     */
    class IndexMaxPQ{
      /**Initializes an empty indexed priority queue with indices between {@code 0}
       * and {@code maxN - 1}.
       * @param {number} maxN the keys on this priority queue are index from {@code 0} to {@code maxN - 1}
       * @param {function} compareFn
       */
      constructor(maxN,compareFn){
        //* @property {number} maxN  maximum number of elements on PQ
        //* @property {number} n number of elements on PQ
        //* @property {array} pq  binary heap using 1-based indexing
        //* @property {array} qp  inverse of pq - qp[pq[i]] = pq[qp[i]] = i
        //* @property {array} mKeys  keys[i] = priority of i
        if(maxN < 0) throw Error("IllegalArgumentException");
        this.compare=compareFn;
        this.maxN = maxN;
        this.n = 0;
        this.mKeys = new Array(maxN + 1); // make this of length maxN??
        this.pq   = new Array(maxN + 1);
        this.qp   = new Array(maxN + 1);  // make this of length maxN??
        for(let i=0; i<=maxN; ++i) this.qp[i] = -1;
      }
      /**Returns true if this priority queue is empty.
       * @return {boolean}
       */
      isEmpty(){
        return this.n == 0;
      }
      /**Is {@code i} an index on this priority queue?
       * @param  {number} i an index
       * @return {boolean}
       */
      contains(i){
        this._validateIndex(i);
        return this.qp[i] != -1;
      }
      /**Returns the number of keys on this priority queue.
       * @return {number}
       */
      size(){
        return this.n;
      }
     /**Associate key with index i.
       * @param {number} i an index
       * @param {any} key the key to associate with index {@code i}
       */
      insert(i, key){
        this._validateIndex(i);
        if(this.contains(i))
          throw Error("index is already in the priority queue");
        ++this.n;
        this.qp[i] = this.n;
        this.pq[this.n] = i;
        this.mKeys[i] = key;
        this._swim(this.n);
      }
      /**Returns an index associated with a maximum key.
       * @return {any}
       */
      maxIndex(){
        if(this.n == 0) throw Error("Priority queue underflow");
        return this.pq[1];
      }
      /**Returns a maximum key.
       * @return {any}
       */
      maxKey(){
        if(this.n == 0) throw Error("Priority queue underflow");
        return this.mKeys[this.pq[1]];
      }
      /**Removes a maximum key and returns its associated index.
       * @return {any}
       */
      delMax(){
        if(this.n == 0) throw Error("Priority queue underflow");
        let max = this.pq[1];
        this._exch(1, this.n--);
        this._sink(1);
        _.assert(this.pq[this.n+1] == max,"bad delMax");
        this.qp[max] = -1;        // delete
        this.mKeys[max] = UNDEF;    // to help with garbage collection
        this.pq[this.n+1] = -1;        // not needed
        return max;
      }
      /**Returns the key associated with index {@code i}.
       * @param  {number} i the index of the key to return
       * @return {any}
       */
      keyOf(i){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        return this.mKeys[i];
      }
      /**Change the key associated with index {@code i} to the specified value.
       * @param  {number} i the index of the key to change
       * @param  {any} key change the key associated with index {@code i} to this key
       */
      changeKey(i, key){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        this.mKeys[i] = key;
        this._swim(this.qp[i]);
        this._sink(this.qp[i]);
      }
      /**Increase the key associated with index {@code i} to the specified value.
       * @param {number} i the index of the key to increase
       * @param {any} key increase the key associated with index {@code i} to this key
       */
      increaseKey(i, key){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        if(this.compare(this.mKeys[i],key) == 0)
          throw Error("Calling increaseKey() with a key equal to the key in the priority queue");
        if(this.compare(this.mKeys[i],key) > 0)
          throw Error("Calling increaseKey() with a key that is strictly less than the key in the priority queue");
        this.mKeys[i] = key;
        this._swim(this.qp[i]);
      }
      /**Decrease the key associated with index {@code i} to the specified value.
       * @param {number} i the index of the key to decrease
       * @param {any} key decrease the key associated with index {@code i} to this key
       */
      decreaseKey(i, key){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        if(this.compare(this.mKeys[i],key) == 0)
          throw Error("Calling decreaseKey() with a key equal to the key in the priority queue");
        if(this.compare(this.mKeys[i],key) < 0)
          throw Error("Calling decreaseKey() with a key that is strictly greater than the key in the priority queue");
        this.mKeys[i] = key;
        this._sink(this.qp[i]);
      }
      /**Remove the key on the priority queue associated with index {@code i}.
       * @param {number} i the index of the key to remove
       */
      delete(i){
        this._validateIndex(i);
        if(!this.contains(i))
          throw Error("index is not in the priority queue");
        let index = this.qp[i];
        this._exch(index, this.n--);
        this._swim(index);
        this._sink(index);
        this.mKeys[i] = UNDEF;
        this.qp[i] = -1;
      }
      _validateIndex(i){
        if(i<0) throw Error("index is negative: " + i);
        if(i>=this.maxN) throw Error("index >= capacity: " + i);
      }
      _less(i,j){
        return less(this.mKeys[this.pq[i]], this.mKeys[this.pq[j]], this.compare)
      }
      _exch(i, j){
        let swap = this.pq[i];
        this.pq[i] = this.pq[j];
        this.pq[j] = swap;
        this.qp[this.pq[i]] = i;
        this.qp[this.pq[j]] = j;
      }
      _swim(k){
        while(k > 1 && this._less(_.ndiv(k,2), k)) {
          this._exch(k, _.ndiv(k,2));
          k = _.ndiv(k,2);
        }
      }
      _sink(k){
        while(2*k <= this.n){
          let j = 2*k;
          if(j < this.n && this._less(j, j+1)) ++j;
          if(!this._less(k, j)) break;
          this._exch(k, j);
          k = j;
        }
      }
      /**Returns an iterator that iterates over the keys.
       * @return {Iterator}
       */
      iter(){
        // add all elements to copy of heap takes linear time since already in heap order so no keys move
        let copy = new IndexMaxPQ(this.pq.length - 1,this.compare);
        for(let i=1; i<=this.n; ++i) copy.insert(this.pq[i], this.mKeys[this.pq[i]]);
        return{
          remove() { throw Error("UnsupportedOperationException")  },
          hasNext() { return !copy.isEmpty() },
          next(){
            if(!this.hasNext()) throw Error("NoSuchElementException");
            return copy.delMax();
          }
        }
      }
      static test(){
        // insert a bunch of strings
        let strings = [ "it", "was", "the", "best", "of", "times", "it", "was", "the", "worst" ];
        let pq = new IndexMaxPQ(strings.length, CMP);
        for(let i=0; i<strings.length; ++i){
          pq.insert(i, strings[i]);
        }
        for(let i,it=pq.iter(); it.hasNext();){
          i=it.next();
          console.log(i + " " + strings[i]);
        }
        console.log("");
        // increase or decrease the key
        for(let i=0; i<strings.length; ++i){
          if(_.rand()<0.5)
            pq.increaseKey(i, strings[i] + strings[i]);
          else
            pq.decreaseKey(i, strings[i].substring(0, 1));
        }
        // delete and print each key
        while(!pq.isEmpty()){
          let key = pq.maxKey();
          let i = pq.delMax();
          console.log(i + " " + key);
        }
        console.log("");
        // reinsert the same strings
        for(let i=0; i<strings.length; ++i){
          pq.insert(i, strings[i]);
        }
        // delete them in random order
        let perm = new Array(strings.length);
        for(let i=0; i<strings.length; ++i) perm[i] = i;
        _.shuffle(perm);
        for(let i=0; i<perm.length; ++i){
          let key = pq.keyOf(perm[i]);
          pq.delete(perm[i]);
          console.log(perm[i] + " " + key);
        }
      }
    }
    //IndexMaxPQ.test();

    const _$={
      FibonacciMinPQ, IndexFibonacciMinPQ,
      Insertion,BinaryInsertion,Selection,Shell,
      Merge,Bubble,Quick,MinPQ, MaxPQ,Heap,IndexMinPQ,IndexMaxPQ
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"))
  }else{
    gscope["io/czlab/mcfud/algo/sort"]=_module
  }

})(this);



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
    const _M= Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();
    if(!Sort) Sort= gscope["io/czlab/mcfud/algo/sort"]();

    const {prnIter, TreeMap,Bag,Stack,Queue,ST,StdCompare:CMP}= Basic;
    const {IndexMinPQ,MinPQ}= Sort;
    const int=Math.floor;
    const {is,u:_}= Core;

    /**
     * @module mcfud/algo_graph
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function _chkVertex(v,V){
      if(v < 0 || v >= V)
        throw Error(`vertex ${v} is not between 0 and ${V-1}`);
      return true;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an undirected graph of vertices named 0 through <em>V</em> – 1.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class Graph{
      /**
       * Initializes an empty graph with {@code V} vertices and 0 edges.
       * param V the number of vertices
       *
       * @param  V number of vertices
       * @throws Error if {@code V < 0}
       */
      constructor(V){
        //* @property {number} V number of vertices
        //* @property {number} E number of edges
        //* @property {array} adjls list of adjacents
        _.assert(V >= 0, "Number of vertices must be non-negative");
        this.verts = V;
        this.edges = 0;
        this.adjls = _.fill(V,()=> new Bag());
      }
      clone(){
        let ret=new Graph(this.V());
        ret.edges= this.E();
        ret.adjls =[];
        for(let v=0,V=this.V(); v<V; ++v)
          ret.adjls.push(this.adjls[v].clone());
        return ret;
      }
      /**Returns the number of vertices in this graph.
       * @return {number}
       */
      V(){
        return this.verts;
      }
      /**Returns the number of edges in this graph.
       * @return {number}
       */
      E(){
        return this.edges;
      }
      /**Adds the undirected edge v-w to this graph.
       * @param  {number} v one vertex in the edge
       * @param  {number} w the other vertex in the edge
       */
      addEdge(v, w){
        _chkVertex(v,this.verts);
        _chkVertex(w,this.verts);
        this.edges+=1;
        this.adjls[v].add(w);
        this.adjls[w].add(v);
      }
      /**Returns the vertices adjacent to vertex {@code v}.
       * @param  {number} v the vertex
       * @return {Bag}
       */
      adj(v){
        return _chkVertex(v, this.verts) && this.adjls[v];
      }
      /**Returns the degree of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      degree(v){
        return _chkVertex(v, this.verts) && this.adjls[v].size();
      }
      /**Returns a string representation of this graph.
       * @return {string}
       */
      toString(){
        let out=`${this.verts} vertices, ${this.edges} edges\n`;
        for(let it,v = 0; v < this.verts; ++v){
          out += `${v}: ` + prnIter(this.adjls[v].iter());
          out += "\n";
        }
        return out;
      }
      static load(V,data){
        let g=new Graph(V);
        _.assert(data.length%2==0,"wanted even n# of data points");
        for(let i=0;i<data.length; i+=2){ g.addEdge(data[i], data[i+1]); }
        return g;
      }
      static test(){
        let obj= Graph.load(13, [0,5,4,3,0,1,9,12,6,4,5,4,0,2,11,12,9,10,0,6,7,8,9,11,5,3]);
        obj.degree(1);
        console.log(obj.toString());
        let c= obj.clone();
        console.log("cloned=\n"+c.toString());
      }
    }
    //Graph.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for determining the vertices
     * connected to a given source vertex <em>s</em>
     *  in an undirected graph.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DepthFirstSearch{
      /**Computes the vertices in graph {@code G} that are
       * connected to the source vertex {@code s}.
       * @param G the graph
       * @param s the source vertex
       */
      constructor(G, s){
        //* @property {array} bMarked marked[v] = is there an s-v path?
        //* @property {number} count number of vertices connected to s
        this.bMarked = new Array(G.V()); // marked[v] = is there an s-v path?
        this.nCount=0; // number of vertices connected to s
        _chkVertex(s,this.bMarked.length) && this._dfs(G, s);
      }
      //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
      // depth first search from v
      _dfs(G, v){
        this.nCount+=1;
        this.bMarked[v] = true;
        for(let w,it= G.adj(v).iter(); it.hasNext();){
          w=it.next();
          if(!this.bMarked[w]) this._dfs(G, w);
        }
      }
      /**Is there a path between the source vertex {@code s} and vertex {@code v}?
       * @param {number} v the vertex
       * @return {boolean}
       */
      marked(v){
        return _chkVertex(v,this.bMarked.length) && this.bMarked[v];
      }
      /**Returns the number of vertices connected to the source vertex {@code s}.
       * @return {number}
       */
      count(){
        return this.nCount;
      }
      static test(){
        let m,obj,g=Graph.load(13,
          [0,5,4,3,0,1,9,12,6,4,5,4,0,2,11,12,9,10,0,6,7,8,9,11,5,3]);
        [0,9].forEach(s=>{
          obj= new DepthFirstSearch(g, s);
          m="";
          for(let v=0; v<g.V(); ++v) if(obj.marked(v)) m+= `${v} `;
          console.log(m);
          console.log(obj.count() != g.V()? "NOT connected" :"connected");
        });
      }
    }
    //DepthFirstSearch.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for finding the vertices connected to a source vertex <em>s</em> in the undirected graph.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class NonrecursiveDFS{
      /**Computes the vertices connected to the source vertex {@code s} in the graph {@code G}.
       * @param {Graph} G the graph
       * @param {number} s the source vertex
       */
      constructor(G, s){
        //* @property {array} bMarked marked[v] = is there an s-v path?
        this.bMarked = new Array(G.V());
        _chkVertex(s,this.bMarked.length);
        // to be able to iterate over each adjacency list, keeping track of which
        // vertex in each adjacency list needs to be explored next
        let adj = _.fill(G.V(),(i)=> G.adj(i).iter());
        // depth-first search using an explicit stack
        let it,v,w,stack = new Stack();
        this.bMarked[s] = true;
        stack.push(s);
        while(!stack.isEmpty()){
          v=stack.peek();
          if(adj[v].hasNext()){
            w = adj[v].next();
            //console.log(`check ${w}`);
            if(!this.bMarked[w]){
              this.bMarked[w] = true;
              stack.push(w);
              //console.log(`dfs(${w})`);
            }
          }else{
            //console.log(`${v} done`);
            stack.pop();
          }
        }
      }
      /**Is vertex {@code v} connected to the source vertex {@code s}?
       * @param {number} v the vertex
       * @return {boolean}
       */
      marked(v){
        return _chkVertex(v,this.bMarked.length) && this.bMarked[v];
      }
      static test(){
        let m,obj,g = Graph.load(13,
          [0,5,4,3,0,1,9,12,6,4,5,4,0,2,11,12,9,10,0,6,7,8,9,11,5,3]);
        [0,9].forEach(s=>{
          obj = new NonrecursiveDFS(g, s);
          m="";
          for(let v=0; v<g.V(); ++v)
            if(obj.marked(v)) m += `${v} `;
          console.log(m);
        })
      }
    }
    //NonrecursiveDFS.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for finding paths from a source vertex <em>s</em>
     * to every other vertex in an undirected graph.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DepthFirstPaths{
      /**Computes a path between {@code s} and every other vertex in graph {@code G}.
       * @param {Graph} G the graph
       * @param {number} s the source vertex
       */
      constructor(G, s){
        //* @property {array} bMarked marked[v] = is there an s-v path?
        //* @property {number} s source index
        //* @property {array} edgeTo edgeTo[v] = last edge on s-v path
        this.bMarked = new Array(G.V());
        this.edgeTo = new Array(G.V());
        this.s = s; // source vertex
        _chkVertex(s,this.bMarked.length) && this._dfs(G, s);
      }
      //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
      // depth first search from v
      _dfs(G, v){
        this.bMarked[v] = true;
        for(let w,it=G.adj(v).iter(); it.hasNext();){
          w=it.next();
          if(!this.bMarked[w]){
            this.edgeTo[w] = v;
            this._dfs(G, w);
          }
        }
      }
      /**Is there a path between the source vertex {@code s} and vertex {@code v}?
       * @param {number} v the vertex
       * @return {boolean}
       */
      hasPathTo(v){
        return _chkVertex(v,this.bMarked.length) && this.bMarked[v];
      }
      /**Returns a path between the source vertex {@code s} and vertex {@code v}, or
       * {@code null} if no such path.
       * @param  {number} v the vertex
       * @return the sequence of vertices on a path between the source vertex
       *         {@code s} and vertex {@code v}, as an Iterable
       */
      pathTo(v){
        if(_chkVertex(v,this.bMarked.length) && this.hasPathTo(v)){
          let path = new Stack();
          for(let x=v; x != this.s; x=this.edgeTo[x]) path.push(x);
          path.push(this.s);
          return path.iter();
        }
      }
      static test(){
        let G = Graph.load(6, [0,5,2,4,2,3,1,2,0,1,3,4,3,5,0,2]);
        let s=0,obj = new DepthFirstPaths(G, s);
        for(let m,it,x, v=0; v<G.V(); ++v){
          if(obj.hasPathTo(v)){
            m= `${s} to ${v}:  `;
            for(let it=obj.pathTo(v); it.hasNext();){
              x=it.next();
              m += x==s? x : `-${x}`;
            }
            console.log(m);
          }else{
            console.log(`${s} to ${v}:  not connected\n`);
          }
        }
      }
    }
    //DepthFirstPaths.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // breadth-first search from a single source
    const _bfs=(G, s, M)=> _bfss(G,[s],M);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // breadth-first search from multiple sources
    function _bfss(G, sources, M){
      let it,v,q = [];
      for(v = 0; v < G.V(); ++v)
        M.mDistTo[v] = Infinity;
      sources.forEach(s=>{
        M.bMarked[s] = true;
        M.mDistTo[s] = 0;
        q.push(s);
      });
      while(q.length>0){
        v=q.shift();
        for(let w,it=G.adj(v).iter(); it.hasNext();){
          w=it.next();
          if(!M.bMarked[w]){
            M.edgeTo[w] = v;
            M.mDistTo[w] = M.mDistTo[v] + 1;
            M.bMarked[w] = true;
            q.push(w);
          }
        }
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // check optimality conditions for single source
    function _check(G, s,M){
      if(M.mDistTo[s] != 0)// check s==0
        throw Error(`dist of source ${s} to itself = ${M.mDistTo[s]}`);
      // each edge v-w dist[w] <= dist[v] + 1
      for(let v = 0; v < G.V(); ++v){
        for(let w, it=G.adj(v).iter(); it.hasNext();){
          w=it.next();
          if(M.hasPathTo(v) !== M.hasPathTo(w)){
            throw Error(`edge ${v}-${w}` +
                        `hasPathTo(${v})=${M.hasPathTo(v)}` +
                        `hasPathTo(${w})=${M.hasPathTo(w)}`);
          }
          if(M.hasPathTo(v) && (M.mDistTo[w] > (M.mDistTo[v]+1))){
            throw Error(`edge ${v}-${w}` +
                        `distTo[${v}]=${M.mDistTo[v]}` +
                        `distTo[${w}]=${M.mDistTo[w]}`);
          }
        }
      }
      // check that v = edgeTo[w] satisfies distTo[w] = distTo[v] + 1
      for(let v,w=0; w<G.V(); ++w){
        if(!M.hasPathTo(w) || w==s){}else{
          v=M.edgeTo[w];
          if(M.mDistTo[w] != M.mDistTo[v]+1){
            throw Error(`shortest path edge ${v}-${w} `+
                        `distTo[${v}]= ${M.mDistTo[v]}`+
                        `distTo[${w}]= ${M.mDistTo[w]}`);
          }
        }
      }
      return true;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function _chkVerts(vs,V){
      if(!vs || vs.length==0)
        throw Error("argument is null or empty");
      vs.forEach(v=> _chkVertex(v,V));
      return true;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for finding shortest paths (number of edges)
     * from a source vertex <em>s</em> (or a set of source vertices)
     * to every other vertex in an undirected graph.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class BreadthFirstPaths{
      /**Computes the shortest path between the source vertex {@code s}
       * and every other vertex in the graph {@code G}.
       * @param {Graph} G the graph
       * @param s {number} the source vertex
       */
      constructor(G, s){
        //* @property {array} bMarked marked[v] = is there an s-v path?
        //* @property {array} mDistTo  number of edges shortest s-v path
        //* @property {array} edgeTo previous edge on shortest s-v path
        this.bMarked = new Array(G.V());
        this.mDistTo = new Array(G.V());
        this.edgeTo = new Array(G.V());
        if(!is.vec(s)){s=[s]}
        _chkVerts(s,G.V());
        _bfss(G, s,this);
        _check(G, s, this);
      }
      /**Is there a path between the source vertex {@code s} (or sources) and vertex {@code v}?
       * @param {number} v the vertex
       * @return {boolean}
       */
      hasPathTo(v){
        return _chkVertex(v,this.bMarked.length) && this.bMarked[v];
      }
      /**Returns the number of edges in a shortest path between the source vertex {@code s}
       * (or sources) and vertex {@code v}?
       * @param {number} v the vertex
       * @return {number}
       */
      distTo(v){
        return _chkVertex(v,this.bMarked.length) && this.mDistTo[v];
      }
      /**Returns a shortest path between the source vertex {@code s} (or sources)
       * and {@code v}, or {@code null} if no such path.
       * @param  {number} v the vertex
       * @return {Iterator}
       */
      pathTo(v){
        if(_chkVertex(v,this.bMarked.length) && this.hasPathTo(v)){
          let x,path = new Stack();
          for(x=v; this.mDistTo[x] != 0; x=this.edgeTo[x]){
            path.push(x);
          }
          path.push(x);
          return path.iter();
        }
      }
      static test(){
        let G=Graph.load(6, [0,5,2,4,2,3,1,2,0,1,3,4,3,5,0,2]);
        //console.log(G.toString());
        let s=0,obj = new BreadthFirstPaths(G, s);
        for(let m,v=0; v<G.V(); ++v){
          if(obj.hasPathTo(v)){
            m=`${s} to ${v}(${obj.distTo(v)}): `;
            for(let x,it=obj.pathTo(v); it.hasNext();){
              x=it.next();
              m += x==s? `${x}`: `-${x}`;
            }
            console.log(m);
          }else{
            console.log(`${s} to ${v} (-):  not connected\n`);
          }
        }
      }
    }
    //BreadthFirstPaths.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a weighted edge in an {@link EdgeWeightedGraph}.
     * Each edge consists of two integers.
     * (naming the two vertices) and a real-value weight.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class Edge{
      /**Initializes an edge between vertices {@code v} and {@code w} of
       * the given {@code weight}.
       * @param  {number} v one vertex
       * @param  {number} w the other vertex
       * @param  {number} weight the weight of this edge
       */
      constructor(v, w, weight){
        //* @property {number} v
        //* @property {number} w
        //* @property {number} weight
        if(v<0) throw Error("vertex index must be a non-negative integer");
        if(w<0) throw Error("vertex index must be a non-negative integer");
        this.v = v;
        this.w = w;
        this._weight = weight;
      }
      /**Returns the weight of this edge.
       * @return {number}
       */
      weight(){
        return this._weight
      }
      /**Returns either endpoint of this edge.
       * @return {number}
       */
      either(){
        return this.v;
      }
      /**Returns the endpoint of this edge that is different from the given vertex.
       * @param  {number} vertex one endpoint of this edge
       * @return {number}
       */
      other(vertex){
        if(vertex == this.v) return this.w;
        if(vertex == this.w) return this.v;
        throw Error("Illegal endpoint");
      }
      /**Compares two edges by weight.
       * @param  {Edge} that the other edge
       * @return {number}
       */
      static comparator(a,b){
        return a._weight<b._weight?-1:(a._weight>b._weight?1:0)
      }
      //compareTo(that){ return this._weight< that._weight?-1:(this._weight>that._weight?1:0) }
      /**Returns a string representation of this edge.
       * @return {string}
       */
      toString(){
        return `${this.v}-${this.w} ${this._weight}`;
      }
      static test(){
        console.log(new Edge(12, 34, 5.67).toString());
      }
    }
    //Edge.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents an edge-weighted graph of vertices named 0 through <em>V</em> – 1,
     * where each undirected edge is of type {@link Edge} and has a real-valued weight.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class EdgeWeightedGraph{
      /**Initializes an empty edge-weighted graph with {@code V} vertices and 0 edges.
       * @param  {number} V the number of vertices
       */
      constructor(V){
        //* @property {number} _V
        //* @property {number} _E
        //* @property {array} adjls
        if(V<0) throw Error("Number of vertices must be non-negative");
        this._V = V;
        this._E = 0;
        this.adjls = _.fill(V,()=> new Bag());
      }
      /**Initializes a random edge-weighted graph with {@code V} vertices and <em>E</em> edges.
       * @param  {number} V the number of vertices
       * @param  {number} E the number of edges
       * @return {Graph}
       */
      static randGraph(V, E){
        let g= new EdgeWeightedGraph(V);
        if(E<0) throw Error("Number of edges must be non-negative");
        for(let wt,v,w,i=0; i<E; ++i){
          v = _.randInt(V);
          w = _.randInt(V);
          wt = Math.round(100 * _.rand()) / 100.0;
          g.addEdge(new Edge(v, w, wt));
        }
        return g;
      }
      /**Initializes a new edge-weighted graph that is a deep copy of {@code G}.
       * @param  {Graph} G the edge-weighted graph to copy
       */
      clone(){
        let g= new EdgeWeightedGraph(this.V());
        g._E = this.E();
        for(let v=0; v<this.V(); ++v)
          g.adjls[v]= this.adjls[v].clone();
        return g;
      }
      /**Returns the number of vertices in this edge-weighted graph.
       * @return {number}
       */
      V(){
        return this._V;
      }
      /**Returns the number of edges in this edge-weighted graph.
       * @return {number}
       */
      E(){
        return this._E;
      }
      /**Adds the undirected edge {@code e} to this edge-weighted graph.
       * @param  {Edge} e the edge
       */
      addEdge(e){
        let v = e.either(),
            w = e.other(v);
        _chkVertex(v,this._V);
        _chkVertex(w,this._V);
        this.adjls[v].add(e);
        this.adjls[w].add(e);
        this._E +=1;
      }
      /**Returns the edges incident on vertex {@code v}.
       * @param  {number} v the vertex
       * @return {Bag}
       */
      adj(v){
        return _chkVertex(v,this._V) && this.adjls[v]
      }
      /**Returns the degree of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      degree(v){
        return _chkVertex(v,this._V) && this.adjls[v].size()
      }
      /**Returns all edges in this edge-weighted graph.
       * To iterate over the edges in this edge-weighted graph, use foreach notation:
       * {@code for (Edge e : G.edges())}.
       * @return {Bag}
       */
      edges(){
        const list = new Bag();
        for(let it,s,e,v=0; v<this._V; ++v){
          s=0;
          for(it=this.adjls[v].iter(); it.hasNext();){
            e=it.next();
            if(e.other(v)>v){
              list.add(e);
            }else if(e.other(v) == v){
              // add only one copy of each self loop (self loops will be consecutive)
              if(s%2 == 0) list.add(e);
              ++s;
            }
          }
        }
        return list.iter();
      }
      /**Returns a string representation of the edge-weighted graph.
       * This method takes time proportional to <em>E</em> + <em>V</em>.
       * @return {string}
       */
      toString(){
        let s = `${this._V} ${this._E}\n`;
        for(let it,v=0; v<this._V; ++v){
          s+= `${v}: `;
          for(it=this.adjls[v].iter(); it.hasNext();){
            s+= `${it.next()}, `;
          }
          s+="\n";
        }
        return s;
      }
      static load(V,data){
        let g= new EdgeWeightedGraph(V);
        _.assert(data.length%3 ==0, "Invalid data size");
        for(let i=0;i<data.length; i+=3){
          _chkVertex(data[i],V) &&
          _chkVertex(data[i+1],V) &&
          g.addEdge(new Edge(data[i],data[i+1], data[i+2]));
        }
        return g;
      }
      static test(){
        let d=`4 5 0.35 4 7 0.37 5 7 0.28 0 7 0.16 1 5 0.32 0 4 0.38 2 3 0.17 1 7 0.19 0 2 0.26 1 2 0.36 1 3 0.29 2 7 0.34 6 2 0.40 3 6 0.52 6 0 0.58 6 4 0.93`.split(" ").map(s=> {return +s});
        let g= EdgeWeightedGraph.load(8,d);
        console.log(g.toString());
      }
    }
    //EdgeWeightedGraph.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for determining the connected components in an undirected graph.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class CC{
      /**Computes the connected components of the undirected graph {@code G}.
       * @param {Graph} undirected or edgeweighted graph
       */
      constructor(G){
        //* @property {array} bMarked   marked[v] = has vertex v been marked?
        //* @property {number} id id[v] = id of connected component containing v
        //* @property {number} size size[id] = number of vertices in given component
        //* @property {number} nCount  number of connected components
        this.bMarked = new Array(G.V());
        this._id = new Array(G.V());
        this._size = new Array(G.V());
        this.nCount=0;
        for(let v=0; v<G.V(); ++v){
          if(!this.bMarked[v]){
            this._dfs(G, v);
            ++this.nCount;
          }
        }
      }
      // depth-first search for a Graph
      _dfs(G, v){
        this.bMarked[v] = true;
        this._id[v] = this.nCount;
        this._size[this.nCount] += 1;
        for(let e,w,it= G.adj(v).iter(); it.hasNext();){
          e=it.next();
          if(G instanceof EdgeWeightedGraph){
            w = e.other(v);
            if(!this.bMarked[w]) this._dfs(G, w);
          }else{
            if(!this.bMarked[e]) this._dfs(G, e);
          }
        }
      }
      /**Returns the component id of the connected component containing vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      id(v){
        return _chkVertex(v,this.bMarked.length) && this._id[v]
      }
      /**Returns the number of vertices in the connected component containing vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      size(v){
        return _chkVertex(v, this.bMarked.length) && this._size[this._id[v]]
      }
      /**Returns the number of connected components in the graph {@code G}.
       * @return {number}
       */
      count(){
        return this.nCount;
      }
      /**Returns true if vertices {@code v} and {@code w} are in the same
       * connected component.
       * @param  {number} v one vertex
       * @param  {number} w the other vertex
       * @return {boolean}
       */
      connected(v, w){
        return _chkVertex(v,this.bMarked.length) &&
               _chkVertex(w, this.bMarked.length) && this.id(v) == this.id(w)
      }
      static test(){
        let G=Graph.load(13, [0,5,4,3,0,1,9,12,6,4,5,4,0,2,11,12,9,10,0,6,7,8,9,11,5,3]);
        //console.log(G.toString());
        let cc=new CC(G);
        // number of connected components
        let m=cc.count();
        console.log(m + " components");
        // compute list of vertices in each connected component
        let cs = _.fill(m, ()=> []);
        for(let v=0; v<G.V(); ++v){
          cs[cc.id(v)].push(v)
        }
        // print results
        for(let s,i=0; i<m; ++i){
          s="";
          cs[i].forEach(v=> s+= v.toString()+" ");
          console.log(s);
        }
      }
    }
    //CC.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a directed graph of vertices
     *  named 0 through <em>V</em> - 1.
     *  It supports the following two primary operations: add an edge to the digraph,
     *  iterate over all of the vertices adjacent from a given vertex.
     *  It also provides
     *  methods for returning the indegree or outdegree of a vertex,
     *  the number of vertices <em>V</em> in the digraph,
     *  the number of edges <em>E</em> in the digraph, and the reverse digraph.
     *  Parallel edges and self-loops are permitted.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class Digraph{
      static load(V,data){
        if(V<0)
          throw Error("verts in a Digraph must be non-negative");
        _.assert(data.length%2==0,"expected even n# of data-length");
        let g= new Digraph(V);
        for(let i=0; i<data.length; i+=2){
          g.addEdge(data[i], data[i+1]);
        }
        return g;
      }
      /**Initializes an empty digraph with <em>V</em> vertices.
       * @param  {number} V the number of vertices
       */
      constructor(V){
        //* @property {array} bMarked   marked[v] = has vertex v been marked?
        //* @property {number} id id[v] = id of connected component containing v
        //* @property {number} size size[id] = number of vertices in given component
        //* @property {number} nCount  number of connected components
        if(V<0) throw Error("verts in a Digraph must be non-negative");
        this._V = V;
        this._E = 0;
        this._indegree = _.fill(V,0);
        this.adjls = _.fill(V,()=> new Bag());
      }
      /**Initializes a new digraph that is a deep copy of the specified digraph.
       * @return {object} digraph copy
       */
      clone(){
        let self=this,
            g=new Digraph(this.V());
        g._E = this.E();
        //update indegrees
        g._indegree = _.fill(g.V(), (i)=> self._indegree[i]);
        // update adjacency lists
        for(let v=0; v<g.V(); ++v){
          g.adjls[v]= this.adjls[v].clone();
        }
        return g;
      }
      /**Returns the number of vertices in this digraph.
       * @return {number}
       */
      V(){
        return this._V;
      }
      /**Returns the number of edges in this digraph.
       * @return {number}
       */
      E(){
        return this._E;
      }
      /**Adds the directed edge v→w to this digraph.
       * @param  {number} v the tail vertex
       * @param  {number} w the head vertex
       */
      addEdge(v, w){
        _chkVertex(v,this._V) && _chkVertex(w,this._V);
        this.adjls[v].add(w);
        this._indegree[w] +=1;
        ++this._E;
      }
      /**Returns the vertices adjacent from vertex {@code v} in this digraph.
       * @param  {number} v the vertex
       * @return {Bag}
       */
      adj(v){
        return _chkVertex(v,this._V) && this.adjls[v]
      }
      /**Returns the number of directed edges incident from vertex {@code v}.
       * This is known as the <em>outdegree</em> of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      outdegree(v){
        return _chkVertex(v,this._V) && this.adjls[v].size()
      }
      /**Returns the number of directed edges incident to vertex {@code v}.
       * This is known as the <em>indegree</em> of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      indegree(v){
        return _chkVertex(v,this._V) && this._indegree[v]
      }
      /**Returns the reverse of the digraph.
       * @return {Digraph}
       */
      reverse(){
        let r= new Digraph(this._V);
        for(let it,v=0; v<this._V; ++v)
          for(it=this.adjls[v].iter(); it.hasNext();){
            r.addEdge(it.next(), v);
          }
        return r;
      }
      /**Returns a string representation of the graph.
       * @return {string}
       */
      toString(){
        let s= `${this._V} vertices, ${this._E} edges\n`;
        for(let it,v=0; v<this._V; ++v){
          s+= `${v}: `
          for(it=this.adjls[v].iter(); it.hasNext();){
            s+= `${it.next()} `
          }
          s+="\n";
        }
        return s;
      }
      static test(){
        let s= "4  2 2  3 3  2 6  0 0  1 2  0 11 12 12  9 9 10 9 11 7  9 10 12 11  4 4  3 3  5 6  8 8  6 5  4 0  5 6  4 6  9 7  6";
        let g= Digraph.load(13, s.split(/\s+/).map(n=>{ return +n }));
        let si="",so="";
        for(let v=0;v<g.V();++v){
          si += `${v}=${g.indegree(v)}, `;
          so += `${v}=${g.outdegree(v)},`;
        }
        console.log("indegreee= "+ si);
        console.log("outdegreee= "+so);
        console.log(g.toString());
        let c= g.clone();
        console.log("cloned=\n"+c.toString());
        let r= g.reverse();
        console.log("rev'ed=\n"+r.toString());
      }
    }
    //Digraph.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for determining the vertices reachable
     * from a given source vertex <em>s</em> (or set of source vertices) in a digraph.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DirectedDFS{
      /**Computes the vertices in digraph {@code G} that are
       * reachable from the source vertex {@code s}.
       * @param {Graph} G the digraph
       * @param {number} s the source vertex
       */
      constructor(G, s){
        //* @property {array} bMarked  marked[v] = true iff v is reachable from source(s)
        //* @property {number} nCount  number of vertices reachable from source(s)
        this.bMarked = new Array(G.V());
        if(!is.vec(s)) s=[s];
        _chkVerts(s,G.V());
        s.forEach(v=>{
          if(!this.bMarked[v]) this._dfs(G, v);
        });
      }
      _dfs(G, v){
        this.mCount+=1;
        this.bMarked[v] = true;
        for(let w,it=G.adj(v).iter(); it.hasNext();){
          w=it.next();
          if(!this.bMarked[w]) this._dfs(G, w);
        }
      }
      /**Is there a directed path from the source vertex (or any
       * of the source vertices) and vertex {@code v}?
       * @param  {number} v the vertex
       * @return {boolean}
       */
      marked(v){
        return _chkVertex(v,this.bMarked.length) && this.bMarked[v]
      }
      /**Returns the number of vertices reachable from the source vertex
       * (or source vertices).
       * @return {number}
       */
      count(){
        return this.mCount;
      }
      static test(){
        let s= "4  2 2  3 3  2 6  0 0  1 2  0 11 12 12  9 9 10 9 11 7  9 10 12 11  4 4  3 3  5 6  8 8  6 5  4 0  5 6  4 6  9 7  6";
        let G= Digraph.load(13, s.split(/\s+/).map(n=>{ return +n }));
        let m="",dfs = new DirectedDFS(G, [1,2,6]);
        // print out vertices reachable from sources
        for(let v = 0; v < G.V(); ++v){
          if(dfs.marked(v)) m+= `${v} `;
        }
        dfs.count();
        console.log(m);
      }
    }
    //DirectedDFS.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for determining whether a digraph has a directed cycle.
     *  The <em>hasCycle</em> operation determines whether the digraph has
     *  a simple directed cycle and, if so, the <em>cycle</em> operation
     *  returns one.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DirectedCycle{
      /**Determines whether the digraph {@code G} has a directed cycle and, if so,
       * finds such a cycle.
       * @param {Graph} G the digraph
       */
      constructor(G){
        //* @property {array} bMarked  marked[v] = has vertex v been marked?
        //* @property {Stack} cycle  directed cycle (or null if no such cycle)
        //* @property {array} edgeTo edgeTo[v] = previous vertex on path to v
        //* @property {array} onStack onStack[v] = is vertex on the stack?
        this.bMarked  = new Array(G.V());
        this.onStack = new Array(G.V());
        this.edgeTo  = new Array(G.V());
        this.mCycle=UNDEF;
        for(let v=0; v<G.V(); ++v)
          if(!this.bMarked[v] && !this.mCycle) this._dfs(G, v);
      }
      // run DFS and find a directed cycle (if one exists)
      _dfs(G, v){
        this.onStack[v] = true;
        this.bMarked[v] = true;
        for(let w, it=G.adj(v).iter(); it.hasNext();){
          w=it.next();
          // short circuit if directed cycle found
          if(this.mCycle){return}
          // found new vertex, so recur
          if(!this.bMarked[w]){
            this.edgeTo[w] = v;
            this._dfs(G, w);
          }else if(this.onStack[w]){
            // trace back directed cycle
            this.mCycle = new Stack();
            for(let x=v; x != w; x=this.edgeTo[x]){
              this.mCycle.push(x);
            }
            this.mCycle.push(w);
            this.mCycle.push(v);
            this._check();
          }
        }
        this.onStack[v] = false;
      }
      /**Does the digraph have a directed cycle?
       * @return {boolean}
       */
      hasCycle(){
        return !!this.mCycle;
      }
      /**Returns a directed cycle if the digraph has a directed cycle, and {@code null} otherwise.
       * @return {Iterator}
       */
      cycle(){
        return this.mCycle && this.mCycle.iter();
      }
      // certify that digraph has a directed cycle if it reports one
      _check(){
        if(this.hasCycle()){
          let first = -1, last = -1;
          for(let v,it=this.cycle(); it.hasNext();){
            v=it.next();
            if(first == -1) first = v;
            last = v;
          }
          if(first != last)
            throw Error(`cycle begins with ${first} and ends with ${last}\n`);
        }
        return true;
      }
      static test(){
        let T2="2 3 0 6 0 1 2 0 11 12  9 12  9 10  9 11 3 5 8 7 5 4 0 5 6 4 6 9 7 6".split(/\s+/).map(n=>{return +n});
        let D=`4  2 2  3 3  2 6  0 0  1 2  0 11 12 12  9 9
               10 9 11 7  9 10 12 11  4 4  3 3  5 6  8 8
               6 5  4 0  5 6  4 6  9 7  6`.split(/\s+/).map(n=>{return +n});
        let s,finder =[new DirectedCycle(Digraph.load(13,D)),
                       new DirectedCycle(Digraph.load(13,T2))];
        finder.forEach(f=>{
          if(f.hasCycle()){
            console.log("Directed cycle: ");
            console.log(prnIter(f.cycle()));
          }else{
            console.log("No directed cycle");
          }
        });
      }
    }
    //DirectedCycle.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a weighted edge in an
     *  {@link EdgeWeightedDigraph}. Each edge consists of two integers
     *  (naming the two vertices) and a real-value weight. The data type
     *  provides methods for accessing the two endpoints of the directed edge and
     *  the weight.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DirectedEdge{
      /**Initializes a directed edge from vertex {@code v} to vertex {@code w} with
       * the given {@code weight}.
       * @param {number} v the tail vertex
       * @param {number} w the head vertex
       * @param {number} weight the weight of the directed edge
       */
      constructor(v, w, weight){
        //* @property {number} v
        //* @property {number} w
        //* @property {number} weight
        if(v<0) throw Error("Vertex names must be non-negative integers");
        if(w<0) throw Error("Vertex names must be non-negative integers");
        this.v = v;
        this.w = w;
        this._weight = weight;
      }
      /**Returns the tail vertex of the directed edge.
       * @return {number}
       */
      from(){
        return this.v;
      }
      /**Returns the head vertex of the directed edge.
       * @return {number}
       */
      to(){
        return this.w;
      }
      /**Returns the weight of the directed edge.
       * @return {number}
       */
      weight(){
        return this._weight;
      }
      /**Returns a string representation of the directed edge.
       * @return {string}
       */
      toString(){
        return `${this.v}->${this.w} ${Number(this._weight).toFixed(2)}`
      }
      static test(){
        console.log(new DirectedEdge(12, 34, 5.67).toString());
      }
    }
    //DirectedEdge.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a edge-weighted digraph of vertices named 0 through <em>V</em> - 1,
     * where each directed edge is of type {@link DirectedEdge} and has a real-valued weight.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class EdgeWeightedDigraph{
      /**Initializes an empty edge-weighted digraph with {@code V} vertices and 0 edges.
       * @param  {number} V the number of vertices
       */
      constructor(V){
        //* @property {number} _V number of vertices in this digraph
        //* @property {number} _E number of edges in this digraph
        //* @property {array} adjls adj[v] = adjacency list for vertex v
        //* @property {array} _indegree  indegree[v] = indegree of vertex v
        if(V<0) throw Error("Number of vertices in a Digraph must be non-negative");
        this._V = V;
        this._E = 0;
        this._indegree = new Array(V);
        this.adjls = _.fill(V,()=> new Bag());
      }
      static randGraph(V, E){
        if (E<0) throw Error("n# edges in a Digraph must be non-negative");
        let g= new EdgeWeightedDigraph(V);
        for(let i=0; i<E; ++i)
          g.addEdge(new DirectedEdge(_.randInt(V),_.randInt(V), 0.01 * _randInt(100)));
        return g;
      }
      static load(V,data){
        if(V<0) throw Error("n# vertices in a Digraph must be non-negative");
        _.assert(data.length%3 ==0, "bad data length");
        let g= new EdgeWeightedDigraph(V);
        for(let i=0; i<data.length; i+=3){
          _chkVertex(data[i],V) &&
          _chkVertex(data[i+1],V) &&
          g.addEdge(new DirectedEdge(data[i],data[i+1],data[i+2]));
          //console.log(`d1=${data[i]}, d2=${data[i+1]}, d3=${data[i+2]}`);
        }
        return g;
      }
      clone(){
        let g=new EdgeWeightedDigraph(this.V());
        g._E = this.E();
        for(let v = 0; v < this.V(); ++v)
          g._indegree[v] = this._indegree(v);
        for(let r,v=0; v<this.V(); ++v){
          g.adjls[v]= this.adjls[v].clone();
        }
        return g;
      }
      /**Returns the number of vertices in this edge-weighted digraph.
       * @return {number}
       */
      V(){
        return this._V;
      }
      /**Returns the number of edges in this edge-weighted digraph.
       * @return {number}
       */
      E(){
        return this._E;
      }
      /**Adds the directed edge {@code e} to this edge-weighted digraph.
       * @param  {DirectedEdge} e the edge
       */
      addEdge(e){
        _.assert(e instanceof DirectedEdge,"Expected DirectedEdge");
        let w = e.to(),
            v = e.from();
        _chkVertex(v,this._V);
        _chkVertex(w,this._V);
        this.adjls[v].add(e);
        this._indegree[w]+=1;
        this._E++;
      }
      /**Returns the directed edges incident from vertex {@code v}.
       * @param  {number} v the vertex
       * @return {Bag}
       */
      adj(v){
        return _chkVertex(v,this._V) && this.adjls[v]
      }
      /**Returns the number of directed edges incident from vertex {@code v}.
       * This is known as the <em>outdegree</em> of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      outdegree(v){
        return _chkVertex(v,this._V) && this.adjls[v].size()
      }
      /**Returns the number of directed edges incident to vertex {@code v}.
       * This is known as the <em>indegree</em> of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      indegree(v){
        return _chkVertex(v,this._V) && this._indegree[v]
      }
      /**Returns all directed edges in this edge-weighted digraph.
       * To iterate over the edges in this edge-weighted digraph, use foreach notation:
       * {@code for (DirectedEdge e : G.edges())}.
       * @return {Iterator}
       */
      edges(){
        const list = new Bag();
        for(let v=0; v<this._V; ++v)
          for(let it= this.adj(v).iter(); it.hasNext();) list.add(it.next());
        return list.iter();
      }
      /**Returns a string representation of this edge-weighted digraph.
       * @return {string}
       */
      toString(){
        let s= `${this._V} ${this._E}\n`;
        for(let v=0; v<this._V; ++v){
          s+= `${v}: ` + prnIter(this.adjls[v].iter()) + "\n";
        }
        return s;
      }
      static test(){
        let data=
        `4 5 0.35
        5 4 0.35
        4 7 0.37
        5 7 0.28
        7 5 0.28
        5 1 0.32
        0 4 0.38
        0 2 0.26
        7 3 0.39
        1 3 0.29
        2 7 0.34
        6 2 0.40
        3 6 0.52
        6 0 0.58
        6 4 0.93`.split(/\s+/).map(n=>{return +n});
        let G = EdgeWeightedDigraph.load(8,data);
        console.log(G.toString());
      }
    }
    //EdgeWeightedDigraph.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for determining depth-first search ordering of the vertices in a digraph
     *  or edge-weighted digraph, including preorder, postorder, and reverse postorder.
     *  <p>
     *  This implementation uses depth-first search.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DepthFirstOrder{
      /**Determines a depth-first order for the digraph {@code G}.
       * @param {Graph} G the digraph
       */
      constructor(G){
        //* @property {array} bMarked marked[v] = has v been marked in dfs?
        //* @property {array} _pre pre[v]    = preorder  number of v
        //* @property {array} _post post[v]   = postorder number of v
        //* @property {array} preorder vertices in preorder
        //* @property {array} postorder vertices in postorder
        //* @property {number} preCounter counter or preorder numbering
        //* @property {number} postCounter counter for postorder numbering
        this._pre= new Array(G.V());
        this._post = new Array(G.V());
        this.preCounter=0;
        this.postCounter=0;
        this.postorder = new Queue();
        this.preorder  = new Queue();
        this.bMarked= new Array(G.V());
        for(let v = 0; v < G.V(); v++)
          if(!this.bMarked[v]) this._dfs(G, v);
        this._check();
      }
      // run DFS in edge-weighted digraph G from vertex v and compute preorder/postorder
      // run DFS in digraph G from vertex v and compute preorder/postorder
      _dfs(G, v){
        this.bMarked[v] = true;
        this._pre[v] = this.preCounter++;
        this.preorder.enqueue(v);
        for(let w, it=G.adj(v).iter(); it.hasNext();){
          w= (G instanceof EdgeWeightedDigraph)? it.next().to() : it.next();
          if(!this.bMarked[w]) this._dfs(G, w);
        }
        this.postorder.enqueue(v);
        this._post[v] = this.postCounter++;
      }
      /**Returns the preorder number of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      pre(v){
        return _chkVertex(v,this.bMarked.length) && this._pre[v]
      }
      /**Returns the postorder number of vertex {@code v}.
       * @param  {number} v the vertex
       * @return {number}
       */
      post(v){
        return _chkVertex(v,this.bMarked.length) && this._post[v]
      }
      /**Returns the vertices in postorder.
       * @return {Iterator}
       */
      postOrder(){
        return this.postorder.iter()
      }
      /**Returns the vertices in preorder.
       * @return {Iterator}
       */
      preOrder(){
        return this.preorder.iter()
      }
      /**Returns the vertices in reverse postorder.
       * @return {Iterator}
       */
      reversePost(){
        let r= new Stack();
        for(let it= this.postorder.iter(); it.hasNext();){
          r.push(it.next())
        }
        return r.iter();
      }
      // check that pre() and post() are consistent with pre(v) and post(v)
      _check(){
        // check that post(v) is consistent with post()
        let it,r = 0;
        for(it= this.postOrder();it.hasNext();){
          if(this.post(it.next()) != r)
            throw Error("post(v) and post() inconsistent");
          ++r;
        }
        // check that pre(v) is consistent with pre()
        r=0;
        for(it=this.preOrder();it.hasNext();){
          if(this.pre(it.next()) != r)
            throw Error("pre(v) and pre() inconsistent");
          ++r;
        }
        return true;
      }
      static test(){
        let G = Digraph.load(13,
                             "2 3 0 6 0 1 2 0 11 12  9 12  9 10  9 11 3 5 8 7 5 4 0 5 6 4 6 9 7 6".split(/\s+/).map(n=>{return +n}));
        console.log(G.toString());
        let s,dfs = new DepthFirstOrder(G);
        console.log("   v  pre  post");
        console.log("--------------");
        for(let v=0; v<G.V(); ++v)
          console.log(`    ${v}  ${dfs.pre(v)}  ${dfs.post(v)}\n`);

        console.log("Preorder:  ");
        console.log(prnIter(dfs.preOrder()));

        console.log("Postorder:  ");
        console.log(prnIter(dfs.postOrder()));
        console.log("");

        console.log("Reverse postorder: ");
        console.log(prnIter(dfs.reversePost()));
      }
    }
    //DepthFirstOrder.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for
     *  determining whether an edge-weighted digraph has a directed cycle.
     *  The <em>hasCycle</em> operation determines whether the edge-weighted
     *  digraph has a directed cycle and, if so, the <em>cycle</em> operation
     *  returns one.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class EdgeWeightedDirectedCycle{
      /**Determines whether the edge-weighted digraph {@code G} has a directed cycle and,
       * if so, finds such a cycle.
       * @param {Graph} G the edge-weighted digraph
       */
      constructor(G){
        //* @property {array} bMarked marked[v] = has v been marked in dfs?
        //* @property {array} edgeTo  edgeTo[v] = previous edge on path to v
        //* @property {array} onStack onStack[v] = is vertex on the stack?
        //* @property {Stack} mCycle directed cycle (or null if no such cycle)
        _.assert(G instanceof EdgeWeightedDigraph,"Expected EdgeWeightedDigraph");
        this.bMarked  = new Array(G.V());
        this.onStack = new Array(G.V());
        this.edgeTo  = new Array(G.V());
        for(let v=0; v<G.V(); ++v)
          if(!this.bMarked[v]) this._dfs(G, v);
        this._check();
      }
      // check that algorithm computes either the topological order or finds a directed cycle
      _dfs(G, v){
        this.onStack[v] = true;
        this.bMarked[v] = true;
        for(let w,e,it=G.adj(v).iter();it.hasNext();){
          e=it.next();
          w= e.to();
          // short circuit if directed cycle found
          if(this.mCycle){return}
          // found new vertex, so recur
          if(!this.bMarked[w]){
            this.edgeTo[w] = e;
            this._dfs(G, w);
          }else if(this.onStack[w]){
            // trace back directed cycle
            this.mCycle = new Stack();
            let f = e;
            while(f.from() != w){
              this.mCycle.push(f);
              f = this.edgeTo[f.from()];
            }
            this.mCycle.push(f);
            return;
          }
        }
        this.onStack[v] = false;
      }
      /**Does the edge-weighted digraph have a directed cycle?
       * @return {boolean}
       */
      hasCycle(){
        return _.echt(this.mCycle)
      }
      /**Returns a directed cycle if the edge-weighted digraph has a directed cycle,
       * and {@code null} otherwise.
       * @return {Iterator}
       */
      cycle(){
        return this.mCycle && this.mCycle.iter()
      }
      // certify that digraph is either acyclic or has a directed cycle
      _check(){
        if(this.hasCycle()){// edge-weighted digraph is cyclic
          let first = UNDEF, last = UNDEF;
          for(let e, it=this.cycle(); it.hasNext();){
            e=it.next();
            if(!first) first = e;
            if(last){
              if(last.to() != e.from())
                throw Error(`cycle edges ${last} and ${e} not incident\n`);
            }
            last = e;
          }
          if(last.to() != first.from())
            throw Error(`cycle edges ${last} and ${first} not incident\n`);
        }
        return true;
      }
      static test(){
        // create random DAG with V vertices and E edges; then add F random edges
        let V = 13,E=8, F=6;
        let G = new EdgeWeightedDigraph(V);
        let vertices = _.shuffle(_.fill(V, (i)=> i));
        for(let wt,v,w,i=0; i<E; ++i){
          do{
            v = _.randInt(V);
            w = _.randInt(V);
          }while(v >= w);
          wt = _.rand();
          G.addEdge(new DirectedEdge(v, w, wt));
        }
        // add F extra edges
        for(let i=0; i<F; ++i){
          G.addEdge(new DirectedEdge(_.randInt(V),_.randInt(V),_.rand()));
        }
        console.log(G.toString());
        // find a directed cycle
        let s,finder = new EdgeWeightedDirectedCycle(G);
        if(finder.hasCycle()){
          console.log("Cycle: " + prnIter(finder.cycle()));
        }else{
          console.log("No directed cycle");
        }
      }
    }
    //EdgeWeightedDirectedCycle.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a digraph, where the
     *  vertex names are arbitrary strings.
     *  By providing mappings between string vertex names and integers,
     *  it serves as a wrapper around the
     *  {@link Digraph} data type, which assumes the vertex names are integers
     *  between 0 and <em>V</em> - 1.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class SymbolGraph{
      /**Initializes a graph from a file using the specified delimiter.
       * Each line in the file contains
       * the name of a vertex, followed by a list of the names
       * of the vertices adjacent to that vertex, separated by the delimiter.
       * @param {array} data 2D array of data
       */
      constructor(data){
        //* @property {ST} st string -> index
        //* @property {array} keys index  -> string
        //* @property {Graph} _graph the underlying digraph
        this.st = new ST();
        // First pass builds the index by reading strings to associate distinct strings with an index
        data.forEach(row=> row.forEach((s,i)=>{
          if(!this.st.contains(s)) this.st.put(s, this.st.size())
        }));
        //inverted index to get string keys in an array
        this.keys = new Array(this.st.size());
        for(let n,it= this.st.keys();it.hasNext();){
          n=it.next();
          this.keys[this.st.get(n)] = n;
        }
        // second pass builds the graph by connecting first vertex on each line to all others
        this._graph = new Graph(this.st.size());
        data.forEach(row=>{
          let v = this.st.get(row[0]);
          for(let w,i=1; i<row.length; ++i){
            w = this.st.get(row[i]);
            this._graph.addEdge(v, w);
          }
        })
      }
      /**Does the graph contain the vertex named {@code s}?
       * @param {number} s the name of a vertex
       * @return {boolean}
       */
      contains(s){
        return this.st.contains(s)
      }
      /**Returns the integer associated with the vertex named {@code s}.
       * @param {number} s the name of a vertex
       * @return {number}
       */
      indexOf(s){
        return this.st.get(s)
      }
      /**Returns the name of the vertex associated with the integer {@code v}.
       * @param  {number} v the integer corresponding to a vertex (between 0 and <em>V</em> - 1)
       * @return {number}
       */
      nameOf(v){
        return _chkVertex(v,this._graph.V()) && this.keys[v]
      }
      /**Returns the graph assoicated with the symbol graph. It is the client's responsibility
       * not to mutate the graph.
       * @return {Graph}
       */
      graph(){
        return this._graph;
      }
      static test(){
        let data=`JFK MCO
                  ORD DEN
                  ORD HOU
                  DFW PHX
                  JFK ATL
                  ORD DFW
                  ORD PHX
                  ATL HOU
                  DEN PHX
                  PHX LAX
                  JFK ORD
                  DEN LAS
                  DFW HOU
                  ORD ATL
                  LAS LAX
                  ATL MCO
                  HOU MCO
                  LAS PHX`.split(/\s+/);
        let sg = new SymbolGraph(_.partition(2,data));
        let graph = sg.graph();
        ["JFK","LAX"].forEach(k=>{
          if(sg.contains(k)){
            let s = sg.indexOf(k);
            console.log(k)
            for(let it= graph.adj(s).iter(); it.hasNext();)
              console.log("   " + sg.nameOf(it.next()));
          }else{
            console.log("input not contain '" + k + "'");
          }
        });
      }
    }
    //SymbolGraph.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a digraph, where the
     *  vertex names are arbitrary strings.
     *  By providing mappings between string vertex names and integers,
     *  it serves as a wrapper around the
     *  {@link Digraph} data type, which assumes the vertex names are integers
     *  between 0 and <em>V</em> - 1.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class SymbolDigraph{
      /**Initializes a digraph from a file using the specified delimiter.
       * Each line in the file contains
       * the name of a vertex, followed by a list of the names
       * of the vertices adjacent to that vertex, separated by the delimiter.
       * @param filename the name of the file
       * @param delimiter the delimiter between fields
       */
      constructor(data){
        //* @property {ST} st string -> index
        //* @property {array} keys index  -> string
        //* @property {Digraph} graph the underlying digraph
        this.st = new ST();
        // First pass builds the index by reading strings to associate
        // distinct strings with an index
        data.forEach(row=> row.forEach(s=>{
          if(!this.st.contains(s))
            this.st.put(s, this.st.size())
        }));
        // inverted index to get string keys in an array
        this.keys = new Array(this.st.size());
        for(let n,it=this.st.keys();it.hasNext();){
          n=it.next();
          this.keys[this.st.get(n)] = n;
        }
        // second pass builds the digraph by connecting first vertex on each
        // line to all others
        this.graph = new Digraph(this.st.size());
        data.forEach(row=> {
          let v = this.st.get(row[0]);
          for(let i=1; i<row.length; ++i)
            this.graph.addEdge(v, this.st.get(row[i]));
        });
      }
      /**Does the digraph contain the vertex named {@code s}?
       * @param s the name of a vertex
       * @return {boolean}
       */
      contains(s){
        return this.st.contains(s)
      }
      /**Returns the integer associated with the vertex named {@code s}.
       * @param s {number} the name of a vertex
       * @return {number}
       */
      indexOf(s){
        return this.st.get(s);
      }
      /**Returns the name of the vertex associated with the integer {@code v}.
       * @param  {number} v the integer corresponding to a vertex (between 0 and <em>V</em> - 1)
       * @return {number} the name of the vertex associated with the integer {@code v}
       */
      nameOf(v){
        return _chkVertex(v, this.graph.V()) && this.keys[v]
      }
      /**Returns the digraph assoicated with the symbol graph. It is the client's responsibility
       * not to mutate the digraph.
       * @return {Digraph}
       */
      digraph(){
        return this.graph;
      }
      static test(){
        let s=`JFK MCO
              ORD DEN
              ORD HOU
              DFW PHX
              JFK ATL
              ORD DFW
              ORD PHX
              ATL HOU
              DEN PHX
              PHX LAX
              JFK ORD
              DEN LAS
              DFW HOU
              ORD ATL
              LAS LAX
              ATL MCO
              HOU MCO
              LAS PHX`.split(/\s+/);
        let sg = new SymbolDigraph(_.partition(2,s));
        let G = sg.digraph();
        ["JFK","ATL","LAX"].forEach(x=>{
          console.log(`${x}`);
          let z=G.adj(sg.indexOf(x)), it=z.iter();
          while(it.hasNext()){
            console.log("   " + sg.nameOf(it.next()));
          }
        });
      }
    }
    //SymbolDigraph.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for
     *  determining a topological order of a <em>directed acyclic graph</em> (DAG).
     *  A digraph has a topological order if and only if it is a DAG.
     *  The <em>hasOrder</em> operation determines whether the digraph has
     *  a topological order, and if so, the <em>order</em> operation
     *  returns one.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class Topological{
      /**Determines whether the digraph {@code G} has a topological order and, if so,
       * finds such a topological order.
       * @param {Graph} G the digraph
       */
      constructor(G){
        this._order=UNDEF;
        this.rank=UNDEF;
        let finder;
        if(G instanceof EdgeWeightedDigraph){
          finder = new EdgeWeightedDirectedCycle(G);
        }else if(G instanceof Digraph){
          finder = new DirectedCycle(G);
          if(!finder.hasCycle()) this.rank = new Array(G.V());
        }else{
          _.assert(false,"bad arg for Topological");
        }
        if(finder && !finder.hasCycle()){
          this._order=new Queue();
          for(let i=0,p,it=new DepthFirstOrder(G).reversePost();it.hasNext();){
            p=it.next();
            if(this.rank)
              this.rank[p] = i++;
            this._order.enqueue(p);
          }
        }
      }
      /**Returns a topological order if the digraph has a topologial order,
       * and {@code null} otherwise.
       * @return {Iterator}
       */
      order(){
        return this._order.iter();
      }
      /**Does the digraph have a topological order?
       * @return {boolean}
       */
      hasOrder(){
        return _.echt(this._order)
      }
      /**The the rank of vertex {@code v} in the topological order;
       * -1 if the digraph is not a DAG
       * @param {number} v the vertex
       * @return {number}
       */
      rank(v){
        return this.rank &&
               _chkVertex(v,this.rank.length) &&
               this.hasOrder()? this.rank[v] : -1;
      }
      static test(){
        let sg = new SymbolDigraph(
        [[`Algorithms`,`Theoretical CS`,`Databases`,`Scientific Computing`],
[`Introduction to CS`,`Advanced Programming`,`Algorithms`],
[`Advanced Programming`,`Scientific Computing`],
[`Scientific Computing`,`Computational Biology`],
[`Theoretical CS`,`Computational Biology`,`Artificial Intelligence`],
[`Linear Algebra`,`Theoretical CS`],
[`Calculus`,`Linear Algebra`],
[`Artificial Intelligence`,`Neural Networks`,`Robotics`,`Machine Learning`],
[`Machine Learning`,`Neural Networks`]]);
        let topological = new Topological(sg.digraph());
        for(let v,it=topological.order(); it.hasNext();){
          console.log(sg.nameOf(it.next()));
        }
      }
    }
    //Topological.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for solving the
     *  single-source shortest paths problem in edge-weighted digraphs
     *  where the edge weights are non-negative.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DepthFirstDirectedPaths{
      /**Computes a directed path from {@code s} to every other vertex in digraph {@code G}.
       * @param  {Graph} G the digraph
       * @param  {number} s the source vertex
       */
      constructor(G, s){
        //* @property {array} marked  marked[v] = true iff v is reachable from s
        //* @property {array} edgeTo  edgeTo[v] = last edge on path from s to v
        //* @property {number} s source vertex
        this.bMarked = new Array(G.V());
        this.edgeTo = new Array(G.V());
        this.s = s;
        _chkVertex(s,this.bMarked.length) && this._dfs(G, s);
      }
      _dfs(G, v){
        this.bMarked[v] = true;
        for(let w,it= G.adj(v).iter();it.hasNext();){
          w=it.next();
          if(!this.bMarked[w]){
            this.edgeTo[w] = v;
            this._dfs(G, w);
          }
        }
      }
      /**Is there a directed path from the source vertex {@code s} to vertex {@code v}?
       * @param  {number} v the vertex
       * @return {boolean}
       */
      hasPathTo(v){
        return _chkVertex(v,this.bMarked.length) && this.bMarked[v];
      }
      /**Returns a directed path from the source vertex {@code s} to vertex {@code v}, or
       * {@code null} if no such path.
       * @param  {number} v the vertex
       * @return {Iterator}
       */
      pathTo(v){
        if(_chkVertex(v,this.bMarked.length) && this.hasPathTo(v)){
          let path = new Stack();
          for(let x=v; x != this.s; x=this.edgeTo[x]) path.push(x);
          path.push(this.s);
          return path.iter();
        }
      }
      static test(){
        let D=`4  2 2  3 3  2 6  0 0  1 2  0 11 12 12  9 9 10
              9 11 7  9 10 12 11  4 4  3 3  5 6
              8 8  6 5  4 0  5 6  4 6  9 7  6`.split(/\s+/).map(n=>{return +n});
        let s=3,G = Digraph.load(13,D);
        let msg, dfs = new DepthFirstDirectedPaths(G, s);
        for(let v=0; v<G.V(); ++v){
          if(dfs.hasPathTo(v)){
            msg= `${s} to ${v}:  `;
            for(let x,it= dfs.pathTo(v);it.hasNext();){
              x=it.next();
              if(x==s) msg += `${x}`;
              else msg += `-${x}`;
            }
            console.log(msg);
          }else{
            console.log(`${s} to ${v}:  not connected`);
          }
        }
      }
    }
    //DepthFirstDirectedPaths.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for solving the
     *  single-source shortest paths problem in edge-weighted digraphs
     *  where the edge weights are non-negative.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class BreadthFirstDirectedPaths{
      /**Computes the shortest path from {@code s} and every other vertex in graph {@code G}.
       * @param {Graph} G the digraph
       * @param {number} s the source vertex
       */
      constructor(G, s){
        //* @property {array} marked marked[v] = is there an s->v path?
        //* @property {array} edgeTo edgeTo[v] = last edge on shortest s->v path
        //* @property {array} distTo distTo[v] = length of shortest s->v path
        if(!is.vec(s)) s= [s];
        this.bMarked = new Array(G.V());
        this.mDistTo = new Array(G.V());
        this.edgeTo = new Array(G.V());
        for(let v=0; v<G.V(); ++v)
          this.mDistTo[v] = Infinity;
        _chkVerts(s,G.V()) && this._bfs(G, s);
      }
      _bfs(G, sources){
        let q = new Queue();
        sources.forEach(s=>{
          this.bMarked[s] = true;
          this.mDistTo[s] = 0;
          q.enqueue(s);
        });
        while(!q.isEmpty()){
          let v=q.dequeue();
          for(let w, it= G.adj(v).iter();it.hasNext();){
            w=it.next();
            if(!this.bMarked[w]){
              this.edgeTo[w] = v;
              this.mDistTo[w] = this.mDistTo[v] + 1;
              this.bMarked[w] = true;
              q.enqueue(w);
            }
          }
        }
      }
      /**Is there a directed path from the source {@code s} (or sources) to vertex {@code v}?
       * @param {number} v the vertex
       * @return {boolean}
       */
      hasPathTo(v){
        return _chkVertex(v,this.bMarked.length) && this.bMarked[v];
      }
      /**Returns the number of edges in a shortest path from the source {@code s}
       * (or sources) to vertex {@code v}?
       * @param {number} v the vertex
       * @return {number}
       */
      distTo(v){
        return _chkVertex(v,this.bMarked.length) && this.mDistTo[v];
      }
      /**Returns a shortest path from {@code s} (or sources) to {@code v}, or
       * {@code null} if no such path.
       * @param {number} v the vertex
       * @return {Iterator}
       */
      pathTo(v){
        if(_chkVertex(v,this.bMarked.length) && this.hasPathTo(v)){
          let x,path = new Stack();
          for(x = v; this.mDistTo[x] != 0; x = this.edgeTo[x]) path.push(x);
          path.push(x);
          return path.iter();
        }
      }
      static test(){
       let D=`4  2 2  3 3  2 6  0 0  1 2  0 11 12 12  9 9 10
              9 11 7  9 10 12 11  4 4  3 3  5 6
              8 8  6 5  4 0  5 6  4 6  9 7  6`.split(/\s+/).map(n=>{return +n});
        let s=3,G = Digraph.load(13,D);
        //console.log(G.toString());
        let msg,bfs = new BreadthFirstDirectedPaths(G,s);
        for(let v=0; v<G.V(); ++v){
          msg="";
          if(bfs.hasPathTo(v)){
            msg= `${s} to ${v} (${bfs.distTo(v)}):  `;
            for(let x,it= bfs.pathTo(v);it.hasNext();){
              x=it.next();
              if(x == s) msg+= `${x}`;
              else msg += `->${x}`;
            }
            console.log(msg);
          }else{
            console.log(`${s} to ${v} (-):  not connected`);
          }
        }
      }
    }
    //BreadthFirstDirectedPaths.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for solving the
     *  single-source shortest paths problem in edge-weighted digraphs
     *  where the edge weights are non-negative.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DijkstraSP{
      /**Computes a shortest-paths tree from the source vertex {@code s} to every other
       * vertex in the edge-weighted digraph {@code G}.
       * @param {Graph} G the edge-weighted digraph
       * @param {number} s the source vertex
       * @param {function} compareFn
       */
      constructor(G, s,compareFn){
        //* @property {array} distTo   distTo[v] = distance  of shortest s->v path
        //* @property {array} edgeTo   edgeTo[v] = last edge on shortest s->v path
        //* @property {IndexMinPQ} pq priority queue of vertices
        _.assert(G instanceof EdgeWeightedDigraph,"Expected EdgeWeightedDigraph");
        for(let e,it=G.edges();it.hasNext();){
          e=it.next();
          if(e.weight() < 0)
            throw Error(`edge ${e} has negative weight`);
        }
        this._distTo = new Array(G.V());
        this.edgeTo = _.fill(G.V(),null);
        _chkVertex(s, G.V());
        for(let v = 0; v < G.V(); ++v)
          this._distTo[v] = Infinity;
        this._distTo[s] = 0;
        // relax vertices in order of distance from s
        this.pq = new IndexMinPQ(G.V(),compareFn);
        this.pq.insert(s, this._distTo[s]);
        while(!this.pq.isEmpty()){
          let v = this.pq.delMin();
          for(let it=G.adj(v).iter(); it.hasNext();)
            this._relax(it.next());
        }
        // check optimality conditions
        this._check(G, s);
      }
      // relax edge e and update pq if changed
      _relax(e){
        let v = e.from(), w = e.to();
        if(this._distTo[w] > this._distTo[v] + e.weight()){
          this._distTo[w] = this._distTo[v] + e.weight();
          this.edgeTo[w] = e;
          if(this.pq.contains(w)) this.pq.decreaseKey(w, this._distTo[w]);
          else this.pq.insert(w, this._distTo[w]);
        }
      }
      /**Returns the length of a shortest path from the source vertex {@code s} to vertex {@code v}.
       * @param  {number} v the destination vertex
       * @return {number}
       */
      distTo(v){
        return _chkVertex(v,this._distTo.length) && this._distTo[v];
      }
      /**Returns true if there is a path from the source vertex {@code s} to vertex {@code v}.
       * @param  {number} v the destination vertex
       * @return {boolean}
       */
      hasPathTo(v){
        return _chkVertex(v,this._distTo.length) && this._distTo[v] < Infinity;
      }
      /**Returns a shortest path from the source vertex {@code s} to vertex {@code v}.
       * @param  {number} v the destination vertex
       * @return {Iterator}
       */
      pathTo(v){
        if(_chkVertex(v,this._distTo.length) && this.hasPathTo(v)){
          let path = new Stack();
          for(let e = this.edgeTo[v]; e; e = this.edgeTo[e.from()])
            path.push(e);
          return path.iter();
        }
      }
      // check optimality conditions:
      // (i) for all edges e:            distTo[e.to()] <= distTo[e.from()] + e.weight()
      // (ii) for all edge e on the SPT: distTo[e.to()] == distTo[e.from()] + e.weight()
      _check(G, s){
        for(let e,it=G.edges();it.hasNext();){
          if(it.next().weight() < 0)
            throw Error("negative edge weight detected");
        }
        // check that distTo[v] and edgeTo[v] are consistent
        if(this._distTo[s] != 0 || this.edgeTo[s])
          throw Error("distTo[s] and edgeTo[s] inconsistent");
        ////
        for(let v=0; v<G.V(); ++v){
          if(v == s) continue;
          if(!this.edgeTo[v] && this._distTo[v] != Infinity)
            throw Error("distTo[] and edgeTo[] inconsistent");
        }
        // check that all edges e = v->w satisfy distTo[w] <= distTo[v] + e.weight()
        for(let v=0; v<G.V(); ++v){
          for(let w,e,it=G.adj(v).iter();it.hasNext();){
            e=it.next();
            w = e.to();
            if(this._distTo[v] + e.weight() < this._distTo[w])
              throw Error(`edge ${e} not relaxed`);
          }
        }
        // check that all edges e = v->w on SPT satisfy distTo[w] == distTo[v] + e.weight()
        for(let v,e,w=0; w<G.V(); ++w){
          if(!this.edgeTo[w]){}else{
            e = this.edgeTo[w];
            v = e.from();
            if(w != e.to()) throw Error("bad edge");
            if(this._distTo[v] + e.weight() != this._distTo[w])
              throw Error(`edge ${e} on shortest path not tight`);
          }
        }
        return true;
      }
      static test(){
        let data= `4 5 0.35
                  5 4 0.35
                  4 7 0.37
                  5 7 0.28
                  7 5 0.28
                  5 1 0.32
                  0 4 0.38
                  0 2 0.26
                  7 3 0.39
                  1 3 0.29
                  2 7 0.34
                  6 2 0.40
                  3 6 0.52
                  6 0 0.58
                  6 4 0.93`.split(/\s+/).map(n=>{return +n});
        let G = EdgeWeightedDigraph.load(8,data);
        //console.log(G.toString());
        let s=0,sp = new DijkstraSP(G, s,CMP);
        // print shortest path
        for(let t=0; t<G.V(); ++t){
          if(sp.hasPathTo(t)){
            console.log(`${s} to ${t} (${Number(sp.distTo(t)).toFixed(2)})  ${prnIter(sp.pathTo(t))}`);
          }else{
            console.log(`${s} to ${t}         no path\n`);
          }
        }
      }
    }
    //DijkstraSP.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function AStarGraphNode(V=0,par=null,g=0,h=0,f=0){
      return{
        parent: par, V, f, g, h,
        equals(o){ return o.V==this.V }
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class AStarSP{
      constructor(G){
        _.assert(G instanceof EdgeWeightedGraph,"Expected EdgeWeightedGraph");
        this.G=G;
      }
      pathTo(start, end, ctx){
        return this._search(this.G,start,end,ctx)
      }
      _search(G,start,end,ctx){
        const CMP=ctx.compare,
              closedSet = new Map(),
              openTM= new Map(),
              openSet = new MinPQ(CMP,10),
              goalNode = AStarGraphNode(end),
              startNode = AStarGraphNode(start),
              rpath=(cn,out)=>{ for(;cn;cn=cn.parent) out.unshift(cn.V); return out; };
        openTM.set(startNode.V,startNode.g);
        openSet.insert(startNode);
        //begin...
        let cur,neighbors=[];
        while(!openSet.isEmpty()){
          cur= openSet.delMin();
          openTM.delete(cur.V);
          closedSet.set(cur.V,0);
          //done?
          if(cur.equals(goalNode)){return rpath(cur,[])}
          //check neigbors
          for(let co,f,g,h,w,it=G.adj(cur.V).iter(); it.hasNext();){
            w=it.next().other(cur.V);
            if(!closedSet.has(w)){
              g = cur.g + ctx.calcCost(w,cur.V);
              h = ctx.calcHeuristic(w,goalNode.V);
              f = g + h;
              //update if lower cost
              if(openTM.has(w) && g > openTM.get(w)){}else{
                openSet.insert(AStarGraphNode(w,cur,g,h,f));
                openTM.set(w, g);
              }
            }
          }
        }
      }
      static test(){
        let D=[0, 1, 111, 0, 2, 85, 1, 3, 104, 1, 4, 140, 1, 5, 183, 2, 3, 230, 2, 6, 67,
               6, 7, 191, 6, 4, 64, 3, 5, 171, 3, 8, 170, 3, 9, 220, 4, 5, 107, 7, 10, 91,
               7, 11, 85, 10, 11, 120, 11, 12, 184, 12, 5, 55, 12, 8, 115, 8, 5, 123,
               8, 9, 189, 8, 13, 59, 13, 14, 81, 9, 15, 102, 14, 15, 126];
        let G= EdgeWeightedGraph.load(16,D);
        let H = {};
        H['7'] = 204;
        H['10'] = 247;
        H['0'] = 215;
        H['6'] = 137;
        H['15'] = 318;
        H['2'] = 164;
        H['8'] = 120;
        H['12'] = 47;
        H['3'] = 132;
        H['9'] = 257;
        H['13'] = 168;
        H['4'] = 75;
        H['14'] = 236;
        H['1'] = 153;
        H['11'] = 157;
        H['5'] = 0;
        let ctx={
          compare(a,b){ return a.f-b.f },
          calcCost(test,cur){
            for(let e,it=G.adj(test).iter();it.hasNext();){
              e=it.next();
              if(e.other(test)==cur) return e.weight();
            }
            throw Error("Boom");
          },
          calcHeuristic(w,g){
            return H[w]
          }
        }
        let c,r,m,p= new AStarSP(G).pathTo(0,5,ctx);
        if(p){
          m=""; p.forEach(n=>{ m+= `[${n}] `; }); console.log(m);
        }else{
          console.log("no path");
        }
      }
    }
    //AStarSP.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Represents a data type for solving
     *  the single-source shortest paths problem in edge-weighted graphs
     *  where the edge weights are non-negative.
     * @memberof module:mcfud/algo_graph
     * @class
     */
    class DijkstraUndirectedSP{
      /**Computes a shortest-paths tree from the source vertex {@code s} to every
       * other vertex in the edge-weighted graph {@code G}.
       * @param {Graph} G the edge-weighted digraph
       * @param {number} s the source vertex
       * @param {function} compareFn
       */
      constructor(G, s,compareFn) {
        _.assert(G instanceof EdgeWeightedGraph,"Expected EdgeWeightedGraph");
        //distTo  distTo[v] = distance  of shortest s->v path
        //edgeTo  edgeTo[v] = last edge on shortest s->v path
        //pq     priority queue of vertices
        for(let e,it=G.edges();it.hasNext();){
          e=it.next();
          if(e.weight()<0)
            throw new Error(`edge ${e} has negative weight`);
        }
        this._distTo = _.fill(G.V(),()=> Infinity);
        this._distTo[s] = 0;
        this.compare=compareFn;
        this.edgeTo = _.fill(G.V(), ()=> null);
        _chkVertex(s,G.V());
        // relax vertices in order of distance from s
        this.pq = new IndexMinPQ(G.V(),this.compare);
        this.pq.insert(s, this._distTo[s]);
        while(!this.pq.isEmpty()){
          let v = this.pq.delMin();
          for(let it=G.adj(v).iter(); it.hasNext();) this._relax(it.next(), v);
        }
        // check optimality conditions
        this._check(G, s);
      }
      // relax edge e and update pq if changed
      _relax(e, v){
        let w = e.other(v);
        if(this._distTo[w] > this._distTo[v] + e.weight()) {
          this._distTo[w] = this._distTo[v] + e.weight();
          this.edgeTo[w] = e;
          if(this.pq.contains(w)) this.pq.decreaseKey(w, this._distTo[w]);
          else this.pq.insert(w, this._distTo[w]);
        }
      }
      /**Returns the length of a shortest path between the source vertex {@code s} and
       * vertex {@code v}.
       * @param  {number} v the destination vertex
       * @return {number}
       */
      distTo(v){
        return _chkVertex(v,this._distTo.length) && this._distTo[v]
      }
      /**Returns true if there is a path between the source vertex {@code s} and
       * vertex {@code v}.
       * @param  {number} v the destination vertex
       * @return {boolean}
       */
      hasPathTo(v){
        return _chkVertex(v,this._distTo.length) && this._distTo[v] < Infinity
      }
      /**Returns a shortest path between the source vertex {@code s} and vertex {@code v}.
       * @param  {number} v the destination vertex
       * @return {Iterator}
       */
      pathTo(v){
        if(_chkVertex(v,this._distTo.length) && this.hasPathTo(v)){
          let x=v,path = new Stack();
          for(let e = this.edgeTo[v]; e; e = this.edgeTo[x]){
            path.push(e);
            x = e.other(x);
          }
          return path.iter();
        }
      }
      // check optimality conditions:
      // (i) for all edges e = v-w:            distTo[w] <= distTo[v] + e.weight()
      // (ii) for all edge e = v-w on the SPT: distTo[w] == distTo[v] + e.weight()
      _check(G, s){
        // check that edge weights are non-negative
        for(let it=G.edges();it.hasNext();){
          if(it.next().weight() < 0)
            throw Error("negative edge weight detected");
        }
        // check that distTo[v] and edgeTo[v] are consistent
        if(this._distTo[s] != 0 || this.edgeTo[s]){
          throw Error("distTo[s] and edgeTo[s] inconsistent");
        }
        for(let v=0; v<G.V(); ++v){
          if(v == s) continue;
          if(!this.edgeTo[v] &&
             this._distTo[v] != Infinity){
            throw Error("distTo[] and edgeTo[] inconsistent");
          }
        }
        // check that all edges e = v-w satisfy distTo[w] <= distTo[v] + e.weight()
        for(let v=0; v<G.V(); ++v){
          for(let w,e,it=G.adj(v).iter();it.hasNext();){
            e=it.next();
            w = e.other(v);
            if(this._distTo[v] + e.weight() < this._distTo[w]){
              throw Error(`edge ${e} not relaxed`);
            }
          }
        }
        // check that all edges e = v-w on SPT satisfy distTo[w] == distTo[v] + e.weight()
        for(let v,e,w=0; w<G.V(); ++w){
          if(!this.edgeTo[w]) continue;
          e = this.edgeTo[w];
          if(w != e.either() && w != e.other(e.either())) return false;
          v = e.other(w);
          if(this._distTo[v] + e.weight() != this._distTo[w]) {
            throw Error(`edge ${e} on shortest path not tight`);
          }
        }
        return true;
      }
      static test(){
        let data=`4 5 0.35 4 7 0.37 5 7 0.28 0 7 0.16 1 5 0.32 0 4 0.38
                  2 3 0.17 1 7 0.19 0 2 0.26 1 2 0.36 1 3 0.29 2 7 0.34
                  6 2 0.40 3 6 0.52 6 0 0.58 6 4 0.93`.split(/\s+/).map(n=>{return +n});
        let s=6,G = EdgeWeightedGraph.load(8,data);
        let sp = new DijkstraUndirectedSP(G, s,CMP);
        for(let m,t=0; t<G.V(); ++t){
          if(sp.hasPathTo(t)){
            m= `${s} to ${t} (${Number(sp.distTo(t)).toFixed(2)})  `;
            for(let it= sp.pathTo(t);it.hasNext();){
              m += `${it.next()}   `;
            }
            console.log(m);
          }else{
            console.log(`${s} to ${t}         no path`);
          }
        }
      }
    }
    //DijkstraUndirectedSP.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      DepthFirstDirectedPaths,
      BreadthFirstDirectedPaths,
      SymbolGraph,
      DijkstraUndirectedSP,
      DijkstraSP,
      Topological,
      SymbolDigraph,
      EdgeWeightedDirectedCycle,
      DepthFirstOrder,
      EdgeWeightedDigraph,
      DirectedEdge,
      DirectedCycle,
      DirectedDFS,
      Digraph,
      CC,
      EdgeWeightedGraph,
      Edge,
      BreadthFirstPaths,
      DepthFirstPaths,
      NonrecursiveDFS,
      DepthFirstSearch,
      Graph
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@/czlab/mcfud"), require("./sort"))
  }else{
    gscope["io/czlab/mcfud/algo/graph"]=_module
  }

})(this);



/* Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright © 2025, Kenneth Leung. All rights reserved. */

;(function(gscope,UNDEF){

	"use strict";

	/**Create the module.
   */
  function _module(Core){

    if(!Core) Core=gscope["io/czlab/mcfud/core"]();
    const int=Math.floor;
    const {u:_, is}= Core;

		/**
     * @module mcfud/algo/NNetGA
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		//Activation Functions
    //For binary classification
		//Use the sigmoid activation function in the output layer.
		//It will squash outputs between 0 and 1, representing
		//probabilities for the two classes.
		//
		//For multi-class classification
		//Use the softmax activation function in the output layer.
		//It will output probability distributions over all classes.
		//
		//If unsure Use the ReLU activation function in the hidden layers.
		//ReLU is the most common default activation function and usually a good choice.
		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function _sigmoid(x){ return 1 / (1 + Math.exp(-x)) }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		const Params={

			mutationRate: 0.1,
			crossOverRate: 0.7,
			probTournament: 0.75,

      NUM_HIDDEN: 1,
      BIAS:-1,
      NUM_ELITES:4,
      TOURNAMENT_SIZE :5,
      MAX_PERTURBATION: 0.3,
      ACTIVATION_RESPONSE: 1,
      NEURONS_PER_HIDDEN: 10,

			sigmoid: _sigmoid,

			relu(x){
				return Math.max(0,x)
			},
			XXtanh(x){
				let a=Math.exp(x), b= Math.exp(-x);
				return (a-b)/(a+b);
			},
			tanh(x){
				return 2 * _sigmoid(2 * x) - 1;
			},
			softmax(logits){
				//seems we need to deal with possible large exp(n) value so
				//do this max thingy...
				//to prevent numerical instability, we subtract the maximum
				//value in x from each element before taking the exponential.
				let exps=[],
					  total, biggest = -Infinity;
				logits.forEach(n=> n>biggest ? (biggest=n) : 0);
				total= logits.reduce((acc,n)=>{
					exps.push(Math.exp(n-biggest));
					return acc+ exps.at(-1);
				},0);
				return exps.map(e=> e/total); // the result probabilities
			},
			XXsoftmax(logits){
				/*
				 * softmax(x[i])= e(x[i])/(sum of all e(x[1...n]))
				*/
				_.assert(is.vec(logits), "Expected array param as softmax input.");
				let exps= logits.map(v=> Math.exp(v));
				let sum= exps.reduce((acc,e)=>acc+e,0);
				let probs= exps.map(e=> e/sum);
				return probs;
			},
			softplus(x){
				return Math.log(1+ Math.exp(x))
			}
    };

		/**
		 * @property {number} avgScore
		 * @property {number} totalScore
		 * @property {number} bestScore
		 * @property {number} worstScore
		 * @property {object} alpha
		 * @class
		 */
		class Statistics{

			#averageScore;
			#totalScore;
			#bestScore;
			#worstScore;
			#best;

			get avgScore(){return this.#averageScore}
			set avgScore(s){this.#averageScore=s}

			get totalScore(){return this.#totalScore}
			set totalScore(s){this.#totalScore=s}

			get bestScore(){return this.#bestScore}
			set bestScore(s){this.#bestScore=s}

			get worstScore(){return this.#worstScore}
			set worstScore(s){this.#worstScore=s}

			get alpha(){return this.#best}
			set alpha(s){this.#best=s}

			/**
			 */
			constructor(){
				this.#averageScore=0;
				this.#totalScore=0;
				this.#bestScore=0;
				this.#worstScore=0;
				this.#best=UNDEF;
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @property {number} activation
		 * @property {number} error
		 * @class
		 */
		class Neuron{

			#activation;
			#weights;
			#parent;
			#error;
			#hasBias;

			get activation(){return this.#activation}
			set activation(n){this.#activation=n}

			get error(){return this.#error}
			set error(e){this.#error=e}

			/**
			 * @param {NeuronLayer} layer
			 * @param {number} inputs
			 * @param {boolean} wantBiasNode
			 */
			constructor(layer,inputs,wantBiasNode=true){
				const ws= _.fill(inputs, ()=> _.randMinus1To1());
				if(wantBiasNode)
					ws.push(_.randMinus1To1());
				this.#parent=layer;
				this.#activation=0;
				this.#weights=ws;
				this.#error=0;
				this.#hasBias=wantBiasNode;
			}
			/**
			 * @return {boolean}
			 */
			hasBias(){
				return this.#hasBias
			}
			/**
			 * @return {number}
			 */
			numInputs(){
				return this.#weights.length
			}
			/**
			 * @return {any} undefined if no bias
			 */
			getBias(){
				return this.#hasBias ? this.#weights.at(-1) : undefined;
			}
			/**
			 * @param {any} b
			 */
			setBias(b){
				if(this.#hasBias)
					this.#weights.with(-1, b);
				return this;
			}
			/**
			 * param {number} i index pos
			 * @return {any}
			 */
			getWeight(i){
				return this.#weights[i]
			}
			/**
			 * @param {number} i index pos
			 * @param {any} w
			 */
			setWeight(i,w){
				_.assert(i>=0&&i<this.#weights.length,"bad index into weights");
				this.#weights[i]=w;
				return this;
			}
			/**
			 * @param {function} func
			 * @param {object} target
			 */
			iterWeights(func, target){
				this.#weights.forEach(func, target);
				return this;
			}
			/**
			 * @param {function} func
			 * @param {object} target
			 * @return {any} result of calling func.
			 */
			applyWeights(func, target){
				return target ? func.call(target, this.#weights) : func(this.#weights)
			}
			/**
			 * @param {any[]} inputs
			 * @param {function} actFunc activation function
			 */
			update(inputs,actFunc){
				let last= this.#hasBias ? this.#weights.length-1 : this.#weights.length;
				let sum=0;
				_.assert(inputs.length>=last, "Incompatible input size for neuron update.");
				for(let i=0; i < last; ++i){
					sum += this.#weights[i] * inputs[i];
				}
				if(this.#hasBias)
					sum += this.#weights.at(-1) * Params.BIAS;
				return this.#activation= actFunc(sum/Params.ACTIVATION_RESPONSE);
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @property {number} numNeurons
		 * @property {Neuron[]} neurons
		 * @class
		 */
		class NeuronLayer{

			#numNeurons;
			#neurons;
			#actFunc;

			get numNeurons(){return this.#neurons.length}

			/**
			 * @param {number} numNeurons
			 * @param {number} numInputsPerNeuron
			 * @param {function} actFunc
			 * @param {boolean} wantBiasNodes
			 */
			constructor(numNeurons, numInputsPerNeuron, actFunc, wantBiasNode=true){
				this.#neurons= _.fill(numNeurons,()=> new Neuron(this,numInputsPerNeuron,wantBiasNode));
				this.#actFunc=actFunc;
			}
			/**
			 * @param {number} index
			 * @return the chosen Neuron
			 */
			neuronAt(i){
				return this.#neurons[i]
			}
			/**
			 * @param {function} cb
			 * @param {object} target
			 */
			iterNeurons(cb, target){
				this.#neurons.forEach(cb, target);
				return this;
			}
			/**
			 * @param {function} func
			 * @param {object} target
			 * @return result of calling func(neurons)
			 */
			applyNeurons(func, target){
				return target ? func.call(target, this.#neurons) : func(this.#neurons)
			}
			/**
			 * @param {any[]} inputs
			 * @return {any[]} array of activations
			 */
			update(inputs){
				return this.#neurons.map(u=> u.update(inputs, this.#actFunc))
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @class
		 */
		class NeuralNet{

			#neuronsPerHidden;
			#numHidden;
			#numOutputs;
			#numInputs;
			#numOfWeights;
			#actFunc;
			#layers;

			get numOutputs(){return this.#numOutputs}
			get numInputs(){return this.#numInputs}

			/**
			 * @param {number} inputs
			 * @param {number} outputs
			 * @param {any[]} hidden
			 * @param {function} actFuncOut
			 * @param {boolean} wantBiasNode
			 */
			constructor(inputs, outputs, [numHidden,perHidden,actFunc], actFuncOut=null, wantBiasNode=true){
				actFuncOut = actFuncOut || Params.sigmoid;
				actFunc= actFunc || Params.sigmoid;
				numHidden=numHidden||0;
				perHidden= perHidden || 0;
				//create the layers of the network
				this.#layers=(function(out){
					if(numHidden>0){
						out.push(new NeuronLayer(perHidden, inputs, actFunc,wantBiasNode));
						for(let i=0; i<numHidden-1; ++i)
							out.push(new NeuronLayer(perHidden,perHidden, actFunc, wantBiasNode));
					}
					return _.conj(out, new NeuronLayer(outputs, numHidden>0?perHidden:inputs, actFuncOut, wantBiasNode));
				})([]);

				this.#neuronsPerHidden=perHidden;
				this.#numHidden=numHidden;
				this.#numInputs=inputs;
				this.#numOutputs=outputs;

				this.#numOfWeights=this.#layers.reduce((sum,y)=>{
					return sum + y.applyNeurons((ns)=>ns.reduce((acc,u)=> acc+u.numInputs())) },0);
			}
			/**
			 * @param {function} func
			 * @param {object} target
			 */
			iterLayers(func,target){
				this.#layers.forEach(func, target);
				return this;
			}
			/**
			 * @param {number} i
			 */
			getLayer(i){ return this.#layers[i] }
			/**
			 * @param {number[]} weights
			 */
			putWeights(weights){
				_.assert(weights.length>=this.#numOfWeights,"bad input to putWeights");
				let pos=0;
				this.#layers.forEach(y=> y.iterNeurons(n=> n.iterWeights((w,i,arr)=>{ arr[i]=weights[pos++] })));
			}
			/**
			 * @return {any[]}
			 */
			getWeights(){
				const out=[];
				for(let y,i=0, z=this.#numHidden+1; i<z; ++i){
					y=this.#layers[i];
					y.iterNeurons(n=> n.iterWeights(w=> out.push(w)))
				}
				return out;
			}
			/**
			 * @return {number}
			 */
			getNumberOfWeights(){
				return this.#numOfWeights
			}
			/**Same as update.
			 * @param {any[]}
			 * @return {any[]}
			 */
			feedForward(inputs){
				return this.update(inputs)
			}
			/**
			 * @param {any[]} inputs
			 * @return {any[]}
			 */
			update(inputs){
				_.assert(inputs.length >= this.#numInputs,"invalid input size");
				let out=[];
				this.#layers.forEach((y,i)=>{
					if(i>0)
						inputs=out;
					out=y.update(inputs)
				});
				return _.assert(out.length == this.#numOutputs, "out length incorrect") ? out : [];
			}
			/**
			 * @return {any[]}
			 */
			calcSplitPoints(){
				let pts= [],
						pos = 0;

				this.#layers.forEach(y=> y.iterNeurons(u=>{
					pos += u.numInputs();
					pts.push(pos-1)
				}));

				return pts;
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @property {number} age
		 * @property {any[]} genes
		 * @class
		 */
		class Chromosome{

			#scoreCalcExtra;
			#scoreCalc;
			#genes;
			#age;

			get age(){return this.#age}
			set age(a){ this.#age=a; }

			/**
			 * @param {any[]} genes
			 * @param {func} scoreCalculator
			 * @param {object} scoreCalcExtra
			 */
			constructor(genes, scoreCalculator, scoreCalcExtra){
				this.#scoreCalcExtra= scoreCalcExtra;
				this.#scoreCalc=scoreCalculator;
				this.#genes=genes;
				this.#age=0;
			}
			/**
			 * @return {array}
			 */
			getScoreCalcInfo(){
				return [this.#scoreCalc, this.#scoreCalcExtra]
			}
			_genes(){ return this.#genes }
			/**
			 * @param {number} i index
			 * @return {any} gene at index i
			 */
			getGeneAt(i){
				return this.#genes[i]
			}
			/**
			 * @param {Chromosome} other
			 * @return {boolean} true if same size
			 */
			compatible(other){
				return this.size() == other.size()
			}
			/**
			 * @return {number} number of genes
			 */
			size(){
				return this.#genes.length
			}
			/**
			 * @return {any[]} deep copy of our genes
			 */
			copyGenes(){
				let yes, a= this.#genes[0];
				try{
					yes= is.obj(a) && is.fun(a.clone);
				}catch(e){
				}
				return yes ? this.#genes.map(g=> g.clone()) : this.#genes.slice();
			}
			/**
			 * @return {any} fitness score
			 */
			getScore(){
				_.assert(false,"Please implement getScore()");
			}
			cmpScore(s){
				_.assert(false,"Please implement cmpScore()");
			}
			/**
			 * @param {any} s
			 */
			updateScore(s){
				_.assert(false,"Please implement updateScore()")
			}
			/**
			*/
			recalcScore(){
				this.updateScore( this.#scoreCalc(this.#genes, this.#scoreCalcExtra));
			}
			/**
			 * @param {Chromosome} other
			 * @return {number} -1 is less, +1 more, 0 is equal.
			 */
			compareTo(other){
				_.assert(false,"Please implement compareTo()")
			}
			/**
			 * @param {function} func
			 * @param {object} target
			 */
			mutateWith(func, target){
				target ? func.call(target, this.#genes) : func(this.#genes);
				this.recalcScore();
				return this;
			}
			/**Choose two random points and “scramble” the genes located between them.
			 *
			 */
			mutateSM(){
				if(_.rand() < Params.mutationRate){
					let [beg, end] = _.randSpan(this.#genes);
					let start=beg+1,count= end-beg-1;
					if(count==2){
						_.swap(this.#genes,start,beg+2)
					}else if(count>2){
						for(let tmp=_.shuffle(this.#genes.slice(start,end)),k=0,i=start; i<end;++i){
							this.#genes[i]=tmp[k++]
						}
					}
					this.recalcScore();
				}
			}
			/**Select two random points, grab the chunk of chromosome
			 * between them and then insert it back into the chromosome
			 * in a random position displaced from the original.
			 */
			mutateDM(){
				if(_.rand() < Params.mutationRate){
					let [beg, end]= _.randSpan(this.#genes);
					let p,tmp,rem, start=beg+1, N=this.#genes.length, count= end-beg-1;
					if(count>0){
						tmp=this.#genes.slice(start, end);
						rem=this.#genes.slice(0, start).concat(this.#genes.slice(end));
						p=_.randInt(rem.length);
						tmp=rem.slice(0,p).concat(tmp).concat(rem.slice(p));
						_.append(this.#genes,tmp,true);
						_.assert(this.#genes.length==N,"mutateDM error");
					}
					this.recalcScore();
				}
			}
			/**Almost the same as the DM operator, except here only one gene is selected
			 * to be displaced and inserted back into the chromosome.
			 */
			mutateIM(){
				if(_.rand() < Params.mutationRate){
					//choose a gene to move
					let pos=_.randInt(this.#genes.length),
							left,right,N=this.#genes.length,v = this.#genes[pos];
					//remove from the chromosome
					this.#genes.splice(pos,1);
					//move the iterator to the insertion location
					pos = _.randInt(this.#genes.length);
					left=this.#genes.slice(0,pos);
					right=this.#genes.slice(pos);
					_.append(this.#genes,left,true);
					this.#genes.push(v);
					_.append(this.#genes,right);
					_.assert(N==this.#genes.length,"mutateIM error");
					this.recalcScore();
				}
			}
			/**Select two random points and reverse the genes between them.
			*/
			mutateIVM(){
				if(_.rand()<Params.mutationRate){
					let [beg, end]= _.randSpan(this.#genes);
					let tmp, start=beg+1, N=this.#genes.length, count= end-beg-1;
					if(count>1){
						tmp=this.#genes.slice(start,end).reverse();
						for(let k=0, i=start; i<end; ++i){
							this.#genes[i]=tmp[k++];
						}
					}
					_.assert(N==this.#genes.length,"mutateIVM error");
					this.recalcScore();
				}
			}
			/**Select two random points, reverse the order between the two points,
			 * and then displace them somewhere along the length of the original chromosome.
			 * This is similar to performing IVM and then DM using the same start and end points.
			 */
			mutateDIVM(){
				if(_.rand()<Params.mutationRate){
					let [beg, end]= _.randSpan(this.#genes);
					let N=this.#genes.length,
							p,tmp,rem,start=beg+1, count= end-beg-1;
					if(count>0){
						tmp=this.#genes.slice(start,end).reverse();
						rem=this.#genes.slice(0, start).concat(this.#genes.slice(end));
						p=_.randInt(rem.length);
						tmp=rem.slice(0,p).concat(tmp).concat(rem.slice(p));
						_.append(this.#genes,tmp,true);
						_.assert(this.#genes.length==N,"mutateDIVM error");
					}
					this.recalcScore();
				}
			}
			/**
			 * @param {function} func
			 * @param {object} target
			 */
			iterGenes(func, target){
				this.#genes.forEach(func, target);
				return this;
			}
			/**
			 * @param {function} func
			 * @param {object} target
			 * @return {any} result calling func
			 */
			applyGenes(func, target){
				return target ? func.call(target, this.#genes) : func(this.#genes)
			}
			/**
			 * @return {Chromosome}
			 */
			clone(){
				_.assert(false,"Please implement clone()")
			}
			/**Several genes are chosen at random from one parent and
			 * then the order of those selections is imposed on
			 * the respective genes in the other parent.
			 * @param {Chromosome} mum
			 * @param {Chromosome} dad
			 * @return {array} newly crossed over genes [g1, g2]
			 */
			static crossOverOBX(mum,dad){
				let b1=mum.copyGenes(), b2=dad.copyGenes();
				if(_.rand() < Params.crossOverRate && mum !== dad){
					_.assert(mum.compatible(dad), "Chromosomes are not compatible.");
					let len=mum.size(),
							pos=_.toGoldenRatio(len)[1],
							positions=_.shuffle(_.fill(len,(i)=>i)).slice(0,pos).sort(),
							temp=positions.map(p=> mum.getGeneAt(p));
					//so now we have n amount of genes from mum in the temp
					//we can impose their order in dad.
					for(let k=0, i=0; i<b2.length; ++i){
						if(k >= temp.length){k=0}
						temp.find(t=>{
							if(b2[i]==t){
								b2[i]=temp[k++];
								return true;
							}
						})
					}
					//now vice versa, first grab from the same positions in dad
					temp=positions.map(p=> dad.getGeneAt(p));
					//and impose their order in mum
					for(let k=0, i=0; i<b1.length; ++i){
						if(k>=temp.length){k=0}
						temp.find(t=>{
							if(b1[i]==t){
								b1[i] = temp[k++];
								return true;
							}
						})
					}
				}
				return [b1, b2];
			}
			/**Similar to Order-Based CrossOver, but instead of imposing the order of the genes,
			 * this imposes the position.
			 * @param {Chromosome} mum
			 * @param {Chromosome} dad
			 * @return {array} newly crossed over genes [g1, g2]
			 */
			static crossOverPBX(mum, dad){
				let b1,b2,len;
				if(_.rand() > Params.crossOverRate || mum === dad){
					b1 = mum.copyGenes();
					b2 = dad.copyGenes();
				}else{
					_.assert(mum.compatible(dad), "Mismatched size of chromosomes.");
					len=mum.size();
					//initialize the babies with null values so we can tell which positions
					//have been filled later in the algorithm
					b1=_.fill(len, null);
					b2=_.fill(len, null);
					_.shuffle(_.fill(len,(i)=>i)).
						slice(0, _.toGoldenRatio(len)[1]).sort().forEach(i=>{
						b1[i] = mum.getGeneAt(i);
						b2[i] = dad.getGeneAt(i);
					});
					//fill the holes
					b2.forEach((v,i)=>{
						if(v===null){
							let rc= mum.applyGenes(gs=> gs.findIndex(g=>{ if(b2.indexOf(g)<0){ b2[i]=g; return true; }}));
							if(rc<0)//couldn't find a value from mum, reuse dad's
								b2[i]=dad.getGeneAt(i);
						}
					});
					b1.forEach((v,i)=>{
						if(v===null){
							let rc= dad.applyGenes(gs=> gs.findIndex(g=>{ if(b1.indexOf(g)<0){ b1[i]=g; return true; }}));
							if(rc<0)//couldn't find a value from dad, reuse mum's
								b1[i]=mum.getGeneAt(i);
						}
					});
					_.assert(!b1.some(x=> x===null), "crossOverPBX null error");
					_.assert(!b2.some(x=> x===null), "crossOverPBX null error");
				}
				return [b1,b2];
			}
			/**
			 * @param {Chromosome} mum
			 * @param {Chromosome} dad
			 * @return {array} newly crossed over genes [g1, g2]
			 */
			static crossOverRND(mum,dad){
				_.assert(mum.compatible(dad), "Mismatched chromosome sizes");
				let cp,b1,b2,len=mum.size();
				if(_.rand() > Params.crossOverRate || mum===dad){
					b1 = mum.copyGenes();
					b2 = dad.copyGenes();
				}else{
					cp = _.randInt(len);
					b1=[];
					b2=[];
					for(let i=0; i<cp; ++i){
						b1.push(mum.getGeneAt(i));
						b2.push(dad.getGeneAt(i));
					}
					for(let i=cp; i<len; ++i){
						b1.push(dad.getGeneAt(i));
						b2.push(mum.getGeneAt(i));
					}
				}
				return [b1,b2];
			}
			/**Partially matched crossover.
			 * @param {Chromosome} mum
			 * @param {Chromosome} dad
			 * @return {array} newly crossed over genes [g1, g2]
			 */
			static crossOverPMX(mum, dad){
				_.assert(mum.compatible(dad), "Mismatched chromosome sizes");
				let len=mum.size(),
						b1 = mum.copyGenes(),
						b2 = dad.copyGenes();
				if(_.rand() > Params.crossOverRate || mum === dad){}else{
					//first we choose a section of the chromosome
					let [beg,end]=_.randSpan(mum.size());
					//now we iterate through the matched pairs of genes from beg
					//to end swapping the places in each child
					for(let p1,p2,g1,g2,pos=beg; pos<end+1; ++pos){
						//these are the genes we want to swap
						g1 = mum.getGeneAt(pos);
						g2 = dad.getGeneAt(pos);
						if(g1 != g2){
							//find and swap them in b1
							p1 = b1.indexOf(g1);
							p2 = b1.indexOf(g2);
							if(p1>=0 && p2>=0) _.swap(b1, p1,p2);
							//and in b2
							p1 = b2.indexOf(g1);
							p2 = b2.indexOf(g2);
							if(p1>=0 && p2>=0) _.swap(b2, p1,p2);
						}
					}
				}
				return [b1,b2];
			}
			/**
			 * @param {Chromosome} mum
			 * @param {Chromosome} dad
			 * @return {array} newly crossed over genes [g1, g2]
			 */
			static crossOverAtSplits(mum, dad){
				_.assert(mum.compatible(dad), "Mismatched chromosome sizes");
				let b1, b2, len=mum.size();
				if(_.rand() > Params.crossOverRate || mum === dad){
					b1=mum.copyGenes();
					b2=dad.copyGenes();
				}else{
					//determine two crossover points
					let [cp1, cp2]= _.randSpan(mum.size());
					b1=[];
					b2=[];
					//create the offspring
					for(let i=0; i<len; ++i){
						if(i<cp1 || i>=cp2){
							//keep the same genes if outside of crossover points
							b1.push(mum.getGeneAt(i));
							b2.push(dad.getGeneAt(i));
						}else{
							//switch over the belly block
							b1.push(dad.getGeneAt(i));
							b2.push(mum.getGeneAt(i));
						}
					}
				}
				return [b1,b2];
			}
		}

		/**
		 * @class
		 */
		class ChromoNumero extends Chromosome{

			#score;

			constructor(genes, calc, target){
				super(genes, calc, target);
				this.recalcScore();
			}
			getScore(){ return this.#score }
			updateScore(s){ this.#score=s; return this; }
			cmpScore(s){ return this.#score>s ? 1 : (this.#score<s? -1 : 0) }
			clone(){
				let [f,t]= this.getScoreCalcInfo();
				return new ChromoNumero(this.copyGenes(), f, t);
			}
			compareTo(other){
				return this.cmpScore(other.getScore());
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function _markStart(extra,fld="cycles"){
			let s= extra.startTime=_.now();
			extra[fld]=0;
			return s;
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function _markEnd(extra){
			return extra.endTime=_.now();
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function _bisectLeft(arr,e){
			//ascending array
      let a,i=0;
      for(;i<arr.length;++i){
        a=arr[i];
        if(a.getScore() == e.getScore() ||
           e.getScore() < a.getScore()) break;
      }
      return i;
    }

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function _newChild(p1, parents, create, crossOver, mutate){
			let p2, tries=5;
			while(tries--){
				p2= _.randInt(parents.length);
				if(p2!=p1) break;
			}
			let c1=parents[p1],
					c,b1,b2,c2=parents[p2];

			if(crossOver){
				[b1,b2]=crossOver(c1,c2);
			}else{
				b1=c1.copyGenes();
				b2=c2.copyGenes();
			}

			b1= create(b1);
			b2= create(b2);

      if(mutate){
        b1.mutateWith(mutate);
				b1.mutateWith(mutate);
      }

			return b1.compareTo(b2)>=0 ? b1 : b2;
    }

		function _dbgScores(pop){
			let s= pop.map(p=> p.getScore()).join(",");
			console.log(s);
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function _genPop(pop,{ crossOver, create, mutate, cycles }){

			if(is.num(pop))
				return _.fill(pop, ()=> create());

			pop.sort(_.comparator(_.SORT_ASC, (a)=>a.getScore(), (b)=>b.getScore()));

			let vecNewPop= pop.slice(pop.length-Params.NUM_ELITES);
			let stats= _$.calcStats(pop);
			let b1,b2,res,mum,dad;

			while(vecNewPop.length < pop.length){
				if(_.randSign()>0 && Params.TOURNAMENT_SIZE !== undefined){
					mum = _$.tournamentSelection(pop,Params.TOURNAMENT_SIZE);
					dad = _$.tournamentSelection(pop,Params.TOURNAMENT_SIZE);
				}else{
					mum = _$.chromoRoulette(pop,stats.totalScore);
					dad = _$.chromoRoulette(pop,stats.totalScore);
				}
				if(crossOver){
					[b1,b2]= crossOver(mum,dad);
				}else{
					b1=mum.copyGenes();
					b2=dad.copyGenes();
				}

				b1=create(b1);
				b2=create(b2);
				if(mutate){
					b1.mutateWith(mutate);
					b2.mutateWith(mutate);
				}

				vecNewPop.push(b1,b2);
			}
			while(vecNewPop.length > pop.length){ vecNewPop.pop() }
			return vecNewPop;
		}

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function* _getNextStar([start,maxMillis],{
			mutate,create,maxAge, poolSize,crossOver
		})
		{
			let par, bestPar = create();
      yield bestPar;
      let parents = [bestPar],
          history = [bestPar],
          ratio,child,index,pindex,lastParIndex;
			poolSize=poolSize || 1;
			maxAge= maxAge || 50;
      for(let i=0;i<poolSize-1;++i){
        par = create();
				if(par.compareTo(bestPar)>0){
          yield (bestPar = par);
          history.push(par);
        }
        parents.push(par);
      }
      lastParIndex = poolSize - 1;
      pindex = 1;
      while(true){
				if(_.now()-start > maxMillis) yield bestPar;
        pindex = pindex>0? pindex-1 : lastParIndex;
        par = parents[pindex];
        child = _newChild(pindex, parents, create, crossOver, mutate);
				if(par.compareTo(child)>0){
          if(maxAge===undefined){ continue }
          par.age += 1;
					if(maxAge > par.age){ continue }
          index = _bisectLeft(history, child, 0, history.length);
          ratio= index / history.length;
          if(_.rand() < Math.exp(-ratio)){
            parents[pindex] = child;
            continue;
          }
          bestPar.age = 0;
          parents[pindex] = bestPar;
          continue;
        }
				if(! (child.compareTo(par)>0)){
          //same fitness
          child.age = par.age + 1;
          parents[pindex] = child;
          continue;
        }
				//child is better, so replace the parent
				child.age = 0;
				parents[pindex] = child;
				//replace best too?
				if(child.compareTo(bestPar)>0){
          yield (bestPar = child);
          history.push(bestPar);
				}
      }
    }

		/**
		 * @class
		 */
		class NeuralGA{
			#generation;
			#extra;
			#popSize;
			#vecPop;
			constructor(size, {create,mutate,crossOver}){
				this.#extra={create, mutate, crossOver};
				this.#generation=1;
				this.#popSize=size;
				this.#vecPop= _genPop(size, this.#extra);
			}
			curGen(){
				return this.#generation;
			}
			epoch(scores){
				_.assert(scores.length == this.#vecPop.length, "GA::Epoch(scores/ chromosomes mismatch)!");
				this.#vecPop.forEach((p,i)=> p.updateScore(scores[i]));
				this.#vecPop= _genPop(this.#vecPop, this.#extra);
				this.#generation += 1;
				return this.createPhenotypes();
			}
			createPhenotypes(){
				return this.#vecPop;
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		const _$={

			NeuronLayer,
			Neuron,
			NeuralNet,

			ChromoNumero,
			Chromosome,

			NeuralGA,

			/**
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {any} optimal
			 * @param {object} extra
			 * @return {array}
			 */
			runGASearch(optimal,extra){
				let start= _markStart(extra),
						maxCycles=(extra.maxCycles|| 100),
						maxMillis= (extra.maxSeconds || 30) * 1000,
						imp, now, gen= _getNextStar([start,maxMillis],extra);
				while(true){
					imp= gen.next().value;
					now= _markEnd(extra);
					if(now-start > maxMillis){
						now=null;
						break;
					}
					if(imp.cmpScore(optimal)>=0){
						break;
					}
					if(extra.cycles >= maxCycles){
						break;
					}
					extra.cycles += 1;
					//console.log(imp.genes.join(","));
				}
				return [now===null, imp]
			},
			/**
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {number|array} pop
			 * @param {object} extra
			 * @return {array}
			 */
			runGACycle(pop,extra){
				let {maxCycles, targetScore, maxSeconds}=extra;
				let s,now, start= _markStart(extra),
						maxMillis= (maxSeconds || 30) * 1000;
				maxCycles= maxCycles || 100;
				while(true){
					pop= _genPop(pop, extra);
					now= _markEnd(extra);
					//time out?
					if(now-start > maxMillis){
						now=null;
						break;
					}
					//pop.forEach(p=> console.log(p._genes().join("")));
					s=_$.calcStats(pop);
					//matched?
					if(_.echt(targetScore) &&
						 s.bestScore >= targetScore){ break }
					//too many?
					if(extra.cycles>= maxCycles){ break }
					extra.cycles += 1;
				}
				return [now === null, pop];
			},
			/**
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {function} optimizationFunction
			 * @param {function} isImprovement
			 * @param {function} isOptimal
			 * @param {function} getNextFeatureValue
			 * @param {any} initialFeatureValue
			 * @param {object} extra
			 * @return {array} [timeout, best]
			 */
			hillClimb(optimizationFunction, isImprovement,
			          isOptimal, getNextFeatureValue, initialFeatureValue,extra){
				let start= _markStart(extra),
						tout, maxMillis= (extra.maxSeconds || 30) * 1000;
				let child,best = optimizationFunction(initialFeatureValue, extra);
				while(!isOptimal(best)){
					child = optimizationFunction( getNextFeatureValue(best), extra);
					if(isImprovement(best, child)){
						best = child
					}
					if(_.now() -start > maxMillis){
						tout=true;
						//time out
						break;
					}
				}
				_markEnd(extra);
				return [tout, best];
			},
			/**Roulette selection.
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop
			 * @param {number} totalScore
			 * @return {Chromosome}
			*/
			getChromoRoulette(pop, totalScore){
				let sum = 0, slice = _.rand() * totalScore;
				return pop.find(p=>{
					//if the fitness so far > random number return the chromo at this point
					sum += p.getScore();
					return sum >= slice ? true : false;
				});
			},
			/**Roulette selection with probabilities.
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop
			 * @param {number} totalScore
			 * @return {Chromosome}
			 */
			chromoRoulette(pop,totalScore){
				let prev=0, R=_.rand();
				let ps=pop.map(p=>{ return prev= (prev+ p.getScore()/totalScore) });
				for(let i=0;i<ps.length-1;++i)
					if(R >= ps[i] && R <= ps[i+1]) return pop[i]
				return pop[0];
			},
			/**
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop
			 * @param {number} N
			 * @return {Chromosome}
			 */
			tournamentSelectionN(pop,N){
				let chosenOne = 0,
						bestSoFar = -Infinity;
				//Select N members from the population at random testing against
				//the best found so far
				for(let k,s,i=0; i<N; ++i){
					k = _.randInt(pop.length);
					s=pop[k].getScore();
					if(s>bestSoFar){
						chosenOne = k;
						bestSoFar = s;
					}
				}
				return pop[chosenOne];
			},
			/**
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop current generation
			 * @return {Chromosome}
			 */
			tournamentSelection(pop){
				let [g1, g2]= _.randSpan(pop);
				if(_.rand() < Params.probTournament){
					return pop[g1].getScore() > pop[g2].getScore() ? pop[g1] : pop[g2]
				}else{
					return pop[g1].getScore() < pop[g2].getScore() ? pop[g1] : pop[g2]
				}
			},
			/**Calculate statistics on population based on scores.
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop current generation
			 * @return {Statistics}
			 */
			calcStats(pop){
				let best= -Infinity,
						worst= Infinity,
						stats=new Statistics();
				pop.forEach(c=>{
					if(c.getScore() > best){
						best = c.getScore();
						stats.bestScore = best;
						stats.alpha= c;
					}else if(c.getScore() < worst){
						worst = c.getScore();
						stats.worstScore = worst;
					}
					stats.totalScore += c.getScore();
				});
				stats.avgScore = stats.totalScore / pop.length;
				return stats;
			},
			/**This type of fitness scaling sorts the population into ascending
			 * order of fitness and then simply assigns a fitness score based on
			 * its position in the ladder.
			 * (so if a genome ends up last it gets score of zero,
			 * if best then it gets a score equal to the size of the population.
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop current generation
			 * @return {Statistics}
			 */
			fitnessScaleRank(pop){
				pop.sort(_.comparator(_.SORT_ASC, (a)=>a.getScore(), (b)=>b.getScore() ));
				//now assign fitness according to the genome's position on
				//this new fitness 'ladder'
				pop.forEach((p,i)=> p.updateScore(i));
				//recalculate values used in selection
				return _$.calcStats(pop);
			},
			/**Scales the fitness using sigma scaling.
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop current generation
			 * @param {Statistics} stats
			 * @return {array} [sigma, new_stats]
			 */
			fitnessScaleSigma(pop, stats){
				//first iterate through the population to calculate the standard deviation
				let total= pop.reduce((acc,p)=> acc + Math.pow(p.getScore()-stats.avgScore,2),0),
						variance = total/pop.length,
						//standard deviation is the square root of the variance
						sigma = Math.sqrt(variance), s2=2*sigma;
				pop.forEach(p=> p.updateScore((p.getScore()-stats.avgScore)/s2));
				return [sigma, _$.calcStats(pop)];
			},
			/**Applies Boltzmann scaling to a populations fitness scores
			 * The static value Temp is the boltzmann temperature which is
			 * reduced each generation by a small amount.
			 * As Temp decreases the difference spread between the high and
			 * low fitnesses increases.
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {array} pop current generation
			 * @param {number} boltzmannTemp
			 * @return {array} [boltzmannTemp, new_stats]
			 */
			fitnessScaleBoltzmann(pop, boltzmannTemp){
				//reduce the temp a little each generation
				boltzmannTemp -= Parmas.BOLTZMANN_DT;
				if(boltzmannTemp< Parmas.MIN_TEMP) boltzmannTemp = Parmas.MIN_TEMP;
				//iterate through the population to find the average e^(fitness/temp)
				//keep a record of e^(fitness/temp) for each individual
				let expBoltz=[],
						avg= pop.reduce((acc,p,x)=>{
							x=Math.exp(p.getScore() / boltzmannTemp);
							expBoltz.push(x);
							return acc+x;
						},0) / pop.length;
				pop.forEach((p,i)=> p.updateScore(expBoltz[i]/avg));
				return [boltzmannTemp, calcStats(pop)];
			},
			/**
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {object} best
			 * @param {object} extra
			 * @param {boolean} timeOut
			 */
			showBest(best,extra,tout){
        console.log(_.fill(80,"-").join(""));
        console.log("total time: " + _.prettyMillis(extra.endTime-extra.startTime));
				if(tout)
					console.log("time expired");
				console.log("total generations= " + extra.cycles);
        console.log("fitness score= "+ best.getScore());
        console.log("best=" + best.applyGenes((gs)=> gs.join(",")));
        console.log(_.fill(80,"-").join(""));
      },
			/**
			 * @memberof module:mcfud/algo/NNetGA
			 * @param {object} options
			 */
			config(options){
				return _.inject(Params, options)
			}
		};

		return _$;
	}

	//export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud")["Core"])
  }else{
    gscope["io/czlab/mcfud/algo/NNetGA"]=_module
  }

})(this)



/* Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright © 2025, Kenneth Leung. All rights reserved. */

;(function(gscope,UNDEF){

  "use strict";

  /**Create the module.
   */
  function _module(Mcfud){

    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();
    const _M = Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();
    const int=Math.floor;
    const {u:_, is}= Core;

    /**
     * @module mcfud/algo/DQL
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const Params={

      ALPHA: 0.1,
      GAMMA: 0.95,
      MAX_EPSILON: 1.0,
      DECAY_RATE: 0.001,
      MIN_EPSILON: 0.05,

      MAX_STEPS: 250,
      EPISODES: 1000,
      SECS_PER_EPISODE: 30

    };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    function argMax(arr){
      let max= -Infinity, pos= -1;
      arr.forEach((v,i)=>{
        if(v>max){
          max=v;pos=i;
        }
      });
      return [pos, max];
    }

    /**
     * @class
     */
    class QLAgent{
      #maxEpsilon;
      #minEpsilon;
      #alpha;
      #gamma;
      #decayRate;
      #qtable;
      constructor(alpha,gamma,minEpsilon,maxEpsilon,decayRate){
        this.#maxEpsilon=maxEpsilon;
        this.#minEpsilon=minEpsilon;
        this.#decayRate=decayRate;
        this.#alpha=alpha;
        this.#gamma=gamma;
        this.#qtable = new Map();
      }
      #decodeKey(state){
        return (is.str(state) ||
          is.num(state) ||
          is.bool(state)) ? state : JSON.stringify(state);
      }
      #safeGetState(state){
        let key=this.#decodeKey(state);
        if(!this.#qtable.has(key))
          this.#qtable.set(key, new Map());
        return this.#qtable.get(key);
      }
      #safeGetAction(state,action,dft=0){
        let s= this.#safeGetState(state);
        return s.has(action) ? s.get(action) : dft;
      }
      getQValue(state, action){
        return this.#safeGetAction(state,action);
      }
      updateQValue(state, action, nextState, reward){
        let cv = this.getQValue(state,action);
        let m= this.#safeGetState(nextState);
        let ks=m.keys().toArray().sort();
        let nvs= ks.map(a=> this.getQValue(nextState,a));
        let max = nvs.length>0 ? argMax(nvs)[1] : 0;
        // q-learning formula
        let nv= cv + this.#alpha * (reward + this.#gamma * max - cv);
        this.#safeGetState(state).set(action, nv);
      }
      getAction(state, listOfActions){
        if(_.rand() < this.#maxEpsilon)
          return _.randItem(listOfActions);
        //choose action with highest q-value
        let max= -Infinity,
            rs= listOfActions.reduce((acc, a, i)=> {
              i= this.getQValue(state,a);
              acc.push([a, i]);
              if(i> max){ max=i }
              return acc;
            },[]);
        let choices= rs.filter(a=> a[1] == max);
        return _.randItem(choices)[0];
      }
      decayEpsilon(episode){
        this.#maxEpsilon = this.#minEpsilon + (this.#maxEpsilon - this.#minEpsilon) * Math.exp(-this.#decayRate *episode);
        //this.#maxEpsilon = Math.max(0, this.#maxEpsilon - this.#decayRate);
      }
      prnQTableAsObj(){
        let obj={};
        this.#qtable.keys().toArray().sort().forEach(k=>{
          let v, o={}, m= this.#qtable.get(k);
          obj[k]=o;
          m.keys().toArray().sort().forEach(k=>{
            o[k]= m.get(k)
          });
        });
        return JSON.stringify(obj);
      }
      prnQTable(){
        let obj=[];
        this.#qtable.keys().toArray().sort().forEach(k=>{
          let v, m= this.#qtable.get(k);
          m.keys().toArray().sort().forEach(i=>{
            v = m.get(i);
            obj.push(`${k},${i},${v}`);
          });
        });
        return obj.join("\n");
      }
      save(){
        //save state,action,qvalue
        //save to file system...
        return this.prnQTable();
      }
      load(data){
        let m, r, arr= data.split("\n");
        m= new Map();
        arr.forEach(a=>{
          r= a.split(",");
          if(!m.has(r[0]))
            m.set(r[0], new Map());
          n= m.get(r[0]);
          n.set(r[1], r[2]);
        });
        this.#qtable=m;
      }
    }

    /**
     * @class
     */
    class Environment{
      #vars;
      constructor(options){
        this.#vars= Object.freeze(_.inject({}, Params, options));
      }
      getVars(){ return this.#vars }
      reset(){
        _.assert(false, "Please implement reset()");
      }
      actionSpace(){
        _.assert(false, "Please implement actionSpace()");
      }
      getState(){
        _.assert(false, "Please implement getState()");
      }
      step(action){
        _.assert(false, "Please implement step()");
      }
    }

    const _$={
      Environment,
      QLAgent
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"));
  }else{
    gscope["io/czlab/mcfud/algo/DQL"]=_module
  }

})(this)




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
  function _module(Mcfud){

    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();
    const _M = Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();

    const {is,u:_}= Core;

    /**
     * @module mcfud/algo_maze
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // Replace a character at index in a string
    function replaceAt(s, i, r){
      return (i> s.length-1)?s :`${s.substr(0,i)}${r}${s.substr(i+1)}`
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function stringVal(s, i){
      // Get the number value at a specific index in a string (0 or 1)
      return +s.charAt(i);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //original source
    //https://github.com/keesiemeijer/maze-generator
    class Maze1{
      /**
       * @param {number} rows
       * @param {number} cols
       * @param {string} entryType "D"|"H"|"V"
      */
      constructor(rows,cols,entryType="D", options={}){
        let {bias,removeWalls,maxWallsRemove} = options;
        this.bias= bias || "";
        this.removeWalls= removeWalls || 0;
        // Maximum 300 walls can be removed
        this.maxWallsRemove= maxWallsRemove || 300;
        this.ROWS=rows;
        this.COLS=cols;
        this.matrix = [];
        this.entryNodes = this.getEntryNodes(entryType);
      }
      generate(){
        this.getMatrix(this.parseMaze(
          //[visited, nswe]
          _.fill(this.COLS*this.ROWS, ()=> "01111")
        ));
        this.removeMazeWalls();
      }
      toAscii(){
        return ""
      }
      getIO(){
        let X = this.entryNodes.end.gate,
          E= this.entryNodes.start.gate;
        return {start: [E.x, E.y], end: [X.x, X.y]};
      }
      toGrid(){
        let r,out=[];
        this.matrix.forEach((row,y)=> {
          r=[];
          for(let i=0;i<row.length;++i){
            r[i]= row.charAt(i)=="1"?1:0
          }
          out.push(r);
        });
        return out;
      }
      parseMaze(nodes){
        const Position = { n: 1, s: 2, w: 3, e: 4, };
        const Opposite= { n: 2, s: 1, w: 4, e: 3 };
        let max = 0,
          visited = 0,
          moveNodes = [],
          biasCount = 0,
          biasFactor = 3,
          last=nodes.length-1,
          pos= _.randInt(nodes.length);
        if(this.bias){
          if("H" == this.bias){
            biasFactor = (this.COLS/100)>=1 ? _M.ndiv(this.COLS,100) + 2 : 3
          }else if("V" == this.bias){
            biasFactor = (this.ROWS/100)>=1 ? _M.ndiv(this.ROWS,100) + 2 : 3
          }
        }
        // Set start node visited.
        nodes[pos] = replaceAt(nodes[pos], 0, 1);
        let next,dir,dirs;
        while(visited < last){
          ++biasCount;
          ++max;
          next = this.getNeighbours(pos);
          dirs = Object.keys(next).filter(k=>
            -1 != next[k] && !stringVal(nodes[next[k]], 0));
          if(this.bias && (biasCount != biasFactor)){
            dirs = this.biasDirections(dirs);
          }else{
            biasCount = 0;
          }
          if(dirs.length){
            ++visited;
            if(dirs.length>1)
              moveNodes.push(pos);
            dir= dirs[_.randInt(dirs.length)];
            // Update current position
            nodes[pos] = replaceAt(nodes[pos], Position[dir], 0);
            // Set new position
            pos = next[dir];
            // Update next position
            nodes[pos] = replaceAt(nodes[pos], Opposite[dir], 0);
            nodes[pos] = replaceAt(nodes[pos], 0, 1);
          }else{
            if(moveNodes.length==0){ break }
            pos= moveNodes.pop();
          }
        }
        return nodes;
      }
      getMatrix(nodes){
        // Add the complete maze in a matrix where 1 is a wall and 0 is a corridor.
        let row1="",
          row2="",
          N = this.COLS* this.ROWS;
        _.assert(nodes.length == N,"invalid nodes");
        for(let i=0; i<N; ++i){
          row1 += row1.length==0 ? "1" : "";
          row2 += row2.length==0 ? "1" : "";
          if(stringVal(nodes[i], 1)){
            row1 += "11";
            row2 += stringVal(nodes[i], 4)? "01" : "00";
          }else{
            let hasAbove = nodes.hasOwnProperty(i-this.COLS);
            let above = hasAbove && stringVal(nodes[i-this.COLS], 4);
            let hasNext = nodes.hasOwnProperty(i+1);
            let next = hasNext && stringVal(nodes[i+1], 1);
            if(stringVal(nodes[i], 4)){
              row1 += "01";
              row2 += "01";
            }else if(next || above){
              row1 += "01";
              row2 += "00";
            }else{
              row1 += "00";
              row2 += "00";
            }
          }
          if((i+1) % this.COLS == 0){
            this.matrix.push(row1,row2);
            row1 = "";
            row2 = "";
          }
        }
        // Add closing row
        this.matrix.push("1".repeat((this.COLS * 2) + 1));
      }
      getEntryNodes(access){
        let entryNodes = {},
          y = ((this.ROWS * 2) + 1) - 2,
          x = ((this.COLS * 2) + 1) - 2;
        if("D" == access){
          entryNodes.start = { x: 1, y: 1, gate: { x: 0, y: 1 } };
          entryNodes.end = { x, y, gate: { x: x + 1, y } };
        }
        if("H" == access || "V" == access){
          let xy = ("H" == access) ? y : x;
          xy = ((xy - 1) / 2);
          let even = xy % 2 == 0;
          xy = even ? xy + 1 : xy;
          let start_x = ("H" == access) ? 1 : xy;
          let start_y = ("H" == access) ? xy : 1;
          let end_x = ("H" == access) ? x : (even ? start_x : start_x + 2);
          let end_y = ("H" == access) ? (even ? start_y : start_y + 2) : y;
          let startgate = ("H" == access) ? { x: 0, y: start_y } : { x: start_x, y: 0 };
          let endgate = ("H" == access) ? { x: x + 1, y: end_y } : { x: end_x, y: y + 1 };
          entryNodes.start = { x: start_x, y: start_y, gate: startgate };
          entryNodes.end = { x: end_x, y: end_y, gate: endgate };
        }
        return entryNodes;
      }
      biasDirections(dirs){
        let hz = dirs.indexOf("w") != -1 || dirs.indexOf("e") != -1,
          vt = dirs.indexOf("n") != -1 || dirs.indexOf("s") != -1;
        return ("H" == this.bias && hz)?
          dirs.filter(k=> "w" == k || "e" == k)
          : ("V" == this.bias && vt)?
          dirs.filter(k=> "n" == k || "s" == k) : dirs;
      }
      getNeighbours(pos){
        return {
          w: (pos>0 && (pos % this.COLS) != 0) ? pos-1 : -1,
          e: ((pos+1) % this.COLS) != 0 ? pos+1 : -1,
          n: (pos - this.COLS)>=0 ? pos - this.COLS : -1,
          s: ((this.COLS* this.ROWS) > (pos + this.COLS)) ? pos + this.COLS : -1
        }
      }
      removeWall(row, index){
        // Remove wall if possible.
        const evenRow = row % 2 == 0,
          evenIndex = index % 2 == 0,
          wall = stringVal(this.matrix[row], index);
        if(!wall){ return false }
        if(!evenRow && evenIndex){
          // Uneven row and even column
          const hasTop = (row-2 > 0) && (1 == stringVal(this.matrix[row-2], index));
          const hasBottom = (row + 2 < this.matrix.length) && (1 == stringVal(this.matrix[row + 2], index));
          if(hasTop && hasBottom){
            this.matrix[row] = replaceAt(this.matrix[row], index, "0");
            return true;
          }
          if(!hasTop && hasBottom){
            const left = 1 == stringVal(this.matrix[row - 1], index - 1);
            const right = 1 == stringVal(this.matrix[row - 1], index + 1);
            if (left || right) {
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }else if(!hasBottom && hasTop){
            const left = 1 == stringVal(this.matrix[row + 1], index - 1);
            const right = 1 == stringVal(this.matrix[row + 1], index + 1);
            if (left || right) {
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }
        }else if(evenRow && !evenIndex){
          // Even row and uneven column
          const hasLeft = 1 == stringVal(this.matrix[row], index - 2);
          const hasRight = 1 == stringVal(this.matrix[row], index + 2);
          if(hasLeft && hasRight){
            this.matrix[row] = replaceAt(this.matrix[row], index, "0");
            return true;
          }
          if(!hasLeft && hasRight){
            const top = 1 == stringVal(this.matrix[row - 1], index - 1);
            const bottom = 1 == stringVal(this.matrix[row + 1], index - 1);
            if(top || bottom){
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }else if(!hasRight && hasLeft){
            const top = 1 == stringVal(this.matrix[row - 1], index + 1);
            const bottom = 1 == stringVal(this.matrix[row + 1], index + 1);
            if(top || bottom){
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }
        }
      }
      removeMazeWalls(){
        if(this.removeWalls==0 ||
           this.matrix.length==0){ return }
        let min = 1,
          tries = 0,
          max = this.matrix.length - 1,
          row,walls, maxTries = this.maxWallsRemove;
        while(tries < maxTries){
          ++tries;
          // Did we reached the goal
          if(this.wallsRemoved >= this.removeWalls){ break }
          // Get random row from matrix
          let y =_.randInt2(min,max);
          y = (y == max) ? y - 1 : y;
          walls = [];
          row = this.matrix[y];
          // Get walls from random row
          for(let w,i = 0; i < row.length; i++){
            if(i == 0 || i == row.length - 1){ continue }
            if( stringVal(row, i)) walls.push(i);
          }
          // Shuffle walls randomly
          _.shuffle(walls);
          // Try breaking a wall for this row.
          for(let i = 0; i < walls.length; i++){
            if(this.removeWall(y, walls[i])){
              // Wall can be broken
              ++this.wallsRemoved;
              break;
            }
          }
        }
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //original
    //https://weblog.jamisbuck.org/2010/12/27/maze-generation-recursive-backtracking
    class Maze2{
      static DV={n:1,s:2,e:4,w:8};
      static DX = { e: 1, w: -1, n: 0, s: 0 };
      static DY = { e:0, w: 0, n: -1, s: 1 };
      static OPPOSITE= { e: "w", w:"e", n:"s",s:"n" };

      /**
       * @param {number} width
       * @param {number} height
       */
      constructor(width,height){
        this.COLS=width;
        this.ROWS=height;
        this._getEntryNodes();
      }
      _getEntryNodes(){
        let nodes = {},
          y = ((this.ROWS * 2) + 1) - 2,
          x = ((this.COLS * 2) + 1) - 2;
        nodes.start = { x: 1, y: 1, gate: { x: 0, y: 1 } };
        nodes.end = { x, y, gate: { x: x + 1, y } };
        this.entryNodes=nodes;
      }
      getIO(){
        let X = this.entryNodes.end.gate,
          E= this.entryNodes.start.gate;
        return {start: [E.x, E.y], end: [X.x, X.y]};
      }
      generate(){
        this.grid=this._walk(0, 0, _.fill(this.ROWS, ()=> _.fill(this.COLS, 0)))
      }
      canSouth(v){
        return (v & Maze2.DV["s"]) != 0
      }
      canEast(v){
        return (v & Maze2.DV["e"]) != 0
      }
      toAscii(){
        let g=this.grid,
          height= g.length,
          out=[], width= g[0].length;
        for(let s, y=0;y<height;++y){
          s="|";
          for(let x=0;x<width;++x){
            s+= this.canSouth(g[y][x]) ? " " : "_";
            if(this.canEast(g[y][x])){
              s+= this.canSouth(g[y][x] | g[y][x+1]) ? " " : "_";
            }else{
              s+= "|";
            }
          }
          out.push(s);
        }
        out.unshift("_".repeat(out[0].length));
        return out.join("\n");
      }
      _walk(cx, cy, grid){
        let nx,ny,dirs = _.shuffle(["n","s","e","w"]);
        dirs.forEach(d=>{
          nx = cx + Maze2.DX[d];
          ny = cy + Maze2.DY[d];

          if(ny>=0&&ny<grid.length &&
             nx>=0&&nx<grid[ny].length && grid[ny][nx]==0){
            grid[cy][cx] |= Maze2.DV[d];
            grid[ny][nx] |= Maze2.DV[Maze2.OPPOSITE[d]];
            this._walk(nx, ny, grid);
          }
        });
        return grid;
      }
      toGrid(){
        let g=this.grid,
          height=g.length,
          r1,r2,out=[], width=g[0].length;
        for(let k,s, y=0;y<height;++y){
          r1=[];
          r2=[];
          k=0;
          r1[k]=1;
          r2[k]=1;
          for(let x=0;x<width;++x){
            ++k;
            if(this.canSouth(g[y][x])){
              r1[k]=0;
              r2[k]=0;
            }else{
              r1[k]=0;
              r2[k]=1;
            }
            ++k;
            if(this.canEast(g[y][x])){
              r1[k]=0;
              r2[k]=1;
            }else{
              r1[k]=1;
              r2[k]=1;
            }
          }
          out.push(r1,r2);
        }
        out.unshift(_.fill(out[0].length, 1));
        return out;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      Maze1, Maze2
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"))
  }else{
    gscope["io/czlab/mcfud/algo/maze"]=_module
  }

})(this);



/* Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright © 2025, Kenneth Leung. All rights reserved. */

;(function(gscope,UNDEF){

  "use strict";

  /**Creates the module.
   */
  function _module(Core){

    if(!Core) Core=gscope["io/czlab/mcfud/core"]();
    const {u:_}=Core;

    /**
      * @module mcfud/algo/minimax
      */

    /**
     * @memberof module:mcfud/algo/minimax
     * @class
     * @property {any} state
     * @property {any} other
     * @property {any} cur
     */
    class GFrame{
      /**
       * @param {any} cur
       * @param {any} other
       */
      constructor(cur,other){
        this.state= UNDEF;
        this.other=other;
        this.cur=cur;
      }
      /**Make a copy of this.
       * @param {function} cp  able to make a copy of state
       * @return {GFrame}
       */
      clone(cp){
        const f= new GFrame();
        f.state=cp(this.state);
        f.other=this.other;
        f.cur=this.cur;
        return f;
      }
    }

    /**Represents a game board.
     * @memberof module:mcfud/algo/minimax
     * @class
     */
    class GameBoard{
      constructor(){
        this.aiActor=UNDEF;
      }
      /**Get the function that copies a game state.
       * @return {function}
       */
      getStateCopier(){}
      /**Get the first move.
       * @param {GFrame} frame
       * @return {any}
       */
      getFirstMove(frame){}
      /**Get the list of next possible moves.
       * @param {GFrame} frame
       * @return {any[]}
       */
      getNextMoves(frame){}
      /**Calculate the score.
       * @param {GFrame} frame
       * @param {number} depth
       * @param {number} maxDepth
       * @return {number}
       */
      evalScore(frame,depth,maxDepth){}
      /**Check if game is a draw.
       * @param {GFrame} frame
       * @return {boolean}
       */
      isStalemate(frame){}
      /**Check if game is over.
       * @param {GFrame} frame
       * @return {boolean}
       */
      isOver(frame,move){}
      /**Reverse previous move.
       * @param {GFrame} frame
       * @param {any} move
       */
      unmakeMove(frame, move){
        if(!this.undoMove)
          throw Error("Need Implementation");
        this.switchPlayer(frame);
        this.undoMove(frame, move);
      }
      //undoMove(frame, move){}
      //doMove(frame, move){ }
      /**Make a move.
       * @param {GFrame} frame
       * @param {any} move
       */
      makeMove(frame, move){
        if(!this.doMove)
          throw Error("Need Implementation!");
        this.doMove(frame, move);
        this.switchPlayer(frame);
      }
      /**Take a snapshot of current game state.
       * @return {GFrame}
       */
      takeGFrame(){}
      /**Switch to the other player.
       * @param {GFrame} snap
       */
      switchPlayer(snap){
        let t = snap.cur;
        snap.cur= snap.other;
        snap.other= t;
      }
      /**Get the other player.
       * @param {any} pv player
       * @return {any}
       */
      getOtherPlayer(pv){
        if(pv === this.actors[1]) return this.actors[2];
        if(pv === this.actors[2]) return this.actors[1];
      }
      /**Get the current player.
       * @return {any}
       */
      getPlayer(){
        return this.actors[0]
      }
      /**Run the algo and get a move.
       * @param {any} seed
       * @param {any} actor
       * @return {any}  the next move
       */
      run(seed,actor){
        this.getAlgoActor=()=>{ return actor }
        this.syncState(seed,actor);
        let pos= this.getFirstMove();
        if(_.nichts(pos))
          pos= _$.evalMiniMax(this);
        return pos;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //+ve if AI wins
    const _calcScore=(board,game,depth,maxDepth)=> board.evalScore(game,depth,maxDepth);

    /**Implements the Min-Max (alpha-beta) algo.
     * @param {GameBoard} board
     * @param {GFrame} game
     * @param {number} depth
     * @param {number} maxDepth
     * @param {number} alpha
     * @param {number} beta
     * @return {number}
     */
    function _miniMax(board, game, depth,maxDepth, alpha, beta, maxing){
      if(depth==0 || board.isOver(game)){
        return [_calcScore(board,game,depth,maxDepth),UNDEF]
      }
      ///////////
      let state=game,
          copier= board.getStateCopier(),
          openMoves= _.shuffle(board.getNextMoves(game));
      if(maxing){
        let rc,pos,move,
            bestMove=openMoves[0], maxValue = -Infinity;
        for(let i=0; i<openMoves.length; ++i){
          if(!board.undoMove){
            _.assert(copier,"Missing state copier!");
            game=state.clone(copier);
          }
          move=openMoves[i];
          board.makeMove(game, move);
					rc= _miniMax(board, game, depth-1, maxDepth, alpha, beta, !maxing)[0];
          if(board.undoMove)
            board.unmakeMove(game,move);
					alpha = Math.max(rc,alpha);
          if(rc > maxValue){
						maxValue = rc;
						bestMove = move;
          }
					if(beta <= alpha){break}
        }
        return [maxValue,bestMove];
      }else{
			  let rc,pos,move,
            bestMove=openMoves[0], minValue = Infinity;
        for(let i=0; i<openMoves.length; ++i){
          if(!board.undoMove){
            _.assert(copier, "Missing state copier!");
            game=state.clone(copier);
          }
          move=openMoves[i];
          board.makeMove(game, move);
					rc = _miniMax(board, game, depth-1, maxDepth, alpha, beta, !maxing)[0];
          if(board.undoMove)
            board.unmakeMove(game, move);
					beta = Math.min(rc,beta);
          if(rc < minValue){
						minValue = rc;
						bestMove = move;
          }
					if(beta <= alpha){break}
        }
        return [minValue,bestMove];
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      algo: "minimax",
      GFrame,
      GameBoard,
      /**Make a move on the game-board using minimax algo.
       * @memberof module:mcfud/algo/minimax
       * @param {GameBoard} board
       * @return {any} next best move
       */
      evalMiniMax(board){
        const f= board.takeGFrame();
        const d= board.depth;
        let [score, move]= _miniMax(board, f, d,d, -Infinity, Infinity, true);
        if(_.nichts(move))
          console.log(`evalMiniMax: score=${score}, pos= ${move}`);
        return move;
      }
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud")["Core"])
  }else{
    gscope["io/czlab/mcfud/algo/minimax"]=_module
  }

})(this);


/* Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright © 2025, Kenneth Leung. All rights reserved. */

;(function(gscope,UNDEF){

  "use strict";

  /**Creates the module.
   */
  function _module(Core){

    if(!Core) Core=gscope["io/czlab/mcfud/core"]();
    const {u:_}=Core;

    /**
      * @module mcfud/algo/negamax
      */

    /**
     * @memberof module:mcfud/algo/negamax
     * @class
     * @property {any} lastBestMove
     * @property {any} state
     * @property {any} other
     * @property {any} cur
     */
    class GFrame{
      /**
       * @param {any} cur
       * @param {any} other
       */
      constructor(cur,other){
        this.lastBestMove=UNDEF;
        this.state= UNDEF;
        this.other=other;
        this.cur=cur;
      }
      /**Make a copy of this.
       * @param {function} cp  able to make a copy of state
       * @return {GFrame}
       */
      clone(cp){
        const f= new GFrame();
        f.state=cp(this.state);
        f.lastBestMove=this.lastBestMove;
        f.other=this.other;
        f.cur=this.cur;
        return f;
      }
    }

    /**Represents a game board.
     * @memberof module:mcfud/algo/negamax
     * @class
     */
    class GameBoard{
      constructor(){}
      /**Get the function that copies a game state.
       * @return {function}
       */
      getStateCopier(){}
      /**Get the first move.
       * @param {GFrame} frame
       * @return {any}
       */
      getFirstMove(frame){}
      /**Get the list of next possible moves.
       * @param {GFrame} frame
       * @return {any[]}
       */
      getNextMoves(frame){}
      /**Calculate the score.
       * @param {GFrame} frame
       * @return {number}
       */
      evalScore(frame){}
      /**Check if game is a draw.
       * @param {GFrame} frame
       * @return {boolean}
       */
      isStalemate(frame){}
      /**Check if game is over.
       * @param {GFrame} frame
       * @return {boolean}
       */
      isOver(frame){}
      /**Reverse previous move.
       * @param {GFrame} frame
       * @param {any} move
       */
      unmakeMove(frame, move){
        if(!this.undoMove)
          throw Error("Need Implementation");
        this.switchPlayer(frame);
        this.undoMove(frame, move);
      }
      //undoMove(frame, move){}
      //doMove(frame, move){ }
      /**Make a move.
       * @param {GFrame} frame
       * @param {any} move
       */
      makeMove(frame, move){
        if(!this.doMove)
          throw Error("Need Implementation!");
        this.doMove(frame, move);
        this.switchPlayer(frame);
      }
      /**Switch to the other player.
       * @param {GFrame} frame
       */
      switchPlayer(snap){
        let t = snap.cur;
        snap.cur= snap.other;
        snap.other= t;
      }
      /**Get the other player.
       * @param {any} pv player
       * @return {any}
       */
      getOtherPlayer(pv){
        if(pv === this.actors[1]) return this.actors[2];
        if(pv === this.actors[2]) return this.actors[1];
      }
      /**Get the current player.
       * @return {any}
       */
      getPlayer(){
        return this.actors[0]
      }
      /**Take a snapshot of current game state.
       * @return {GFrame}
       */
      takeGFrame(){}
      run(seed,actor){
        this.getAlgoActor=()=>{ return actor }
        this.syncState(seed,actor);
        let pos= this.getFirstMove();
        if(_.nichts(pos))
          pos= _$.evalNegaMax(this);
        return pos;
      }
    }

    /** @ignore */
    function _calcScore(board,game,depth,maxDepth){
      //if the other player wins, then return a -ve else +ve
      //maxer == 1 , minus == -1
      let score=board.evalScore(game,depth,maxDepth);
      /*
      if(!_.feq0(score))
        score -= 0.01*depth*Math.abs(score)/score;
      return score;
      */
      return score * (1 + 0.001 * depth);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //option2
    function _negaAlphaBeta(board, game, depth, maxDepth, alpha, beta){

      if(depth==0 || board.isOver(game)){
        return { depth, value: _calcScore(board,game,depth,maxDepth) }
      }

      let state=game,
          copier= board.getStateCopier(),
          openMoves= _.shuffle(board.getNextMoves(game));

      for(let rc, move, i=0; i< openMoves.length; ++i){
        move= openMoves[i];
        if(!board.undoMove){
          _.assert(copier, "Missing state copier!");
          game= state.clone(copier);
        }
        board.makeMove(game, move);
        rc = _negaAlphaBeta(board, game, depth-1,
                                         maxDepth,
                                         {value: -beta.value, move: beta.move},
                                         {value: -alpha.value, move: alpha.move});
        //now, roll it back
        if(board.undoMove)
          board.unmakeMove(game, move);
        rc.value = -rc.value;
        rc.move = move;
        if(rc.value>alpha.value){
          alpha = {value: rc.value, move: move, depth: rc.depth};
        }
        if(alpha.value >= beta.value){
          return beta;
        }
      }

      return JSON.parse(JSON.stringify(alpha));
    }

    /**Implements the NegaMax Min-Max algo.
     * @see {@link https://github.com/Zulko/easyAI}
     * @param {GameBoard} board
     * @param {GFrame} game
     * @param {number} depth
     * @param {number} maxDepth
     * @param {number} alpha
     * @param {number} beta
     * @return {number}
     */
    function _negaMax(board, game, depth,maxDepth,alpha, beta){

      if(depth==0 || board.isOver(game)){
        return [_calcScore(board,game,depth,maxDepth),null]
      }

      let openMoves = _.shuffle(board.getNextMoves(game)),
          copier= board.getStateCopier(),
          state=game,
          bestValue = -Infinity,
          bestMove = openMoves[0];

      if(depth==maxDepth)
        state.lastBestMove=bestMove;

      for(let rc, move, i=0; i<openMoves.length; ++i){
        if(!board.undoMove){
          _.assert(copier, "Missing state copier!");
          game=state.clone(copier);
        }
        move = openMoves[i];
        //try a move
        board.makeMove(game, move);
        rc= - _negaMax(board, game, depth-1, maxDepth, -beta, -alpha)[0];
        //now, roll it back
        if(board.undoMove)
          board.unmakeMove(game, move);
        //how did we do ?
        if(bestValue < rc){
          bestValue = rc;
          bestMove = move
        }
        if(alpha < rc){
          alpha=rc;
          if(depth == maxDepth)
            state.lastBestMove = move;
          if(alpha >= beta) break;
        }
      }

      return [bestValue, state.lastBestMove];
    }

    const _$={
      algo:"negamax",
      GFrame,
      GameBoard,
      /**Make a move on the game-board using negamax algo.
       * @memberof module:mcfud/algo/negamax
       * @param {GameBoard} board
       * @return {any} next best move
       */
      XXevalNegaMax(board){
        const f= board.takeGFrame();
        const d= board.depth;
        let [score,move]= _negaMax(board, f, d,d, -Infinity, Infinity);
        if(_.nichts(move))
          console.log(`evalNegaMax: score=${score}, pos= ${move}, lastBestMove=${move}`);
        return move;
      },
      evalNegaMax(board){
        const f= board.takeGFrame();
        const d= board.depth;
        let {value, move} = _negaAlphaBeta(board, f, d, d, {value: -Infinity },
                                                           {value: Infinity  });
        if(_.nichts(move))
          console.log(`evalNegaMax: score= ${value}, pos= ${move}`);
        return move;
      }
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud")["Core"])
  }else{
    gscope["io/czlab/mcfud/algo/negamax"]=_module
  }

})(this);


/* Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright © 2025, Kenneth Leung. All rights reserved. */

;(function(gscope,UNDEF){

  "use strict";

  /**Create the module.
   */
  function _module(Mcfud){

    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();
    const _M = Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();
    const int=Math.floor;
    const {u:_, is}= Core;

    /**
     * @module mcfud/algo/NEAT_CBullet
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @typedef {object} NodeType
     * @property {number} INPUT
     * @property {number} HIDDEN
     * @property {number} OUTPUT
     * @property {number} BIAS
     * @property {number} NONE
     */
    const NodeType={ INPUT: 0, OUTPUT: 1, BIAS:2, HIDDEN:3, NONE: 4 };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @typedef {object} LinkType
     * @property {number} NEURON
     * @property {number} LINK
     */
    const LinkType={ NEURON: 1, LINK: 2 };

    ////////////////////////////////////////////////////////////////////////////
    const Params={
      nextInnov: 42,
      BIAS: 1,
      probMutateWeight: 0.1,
      probMutateLink: 0.8,
      probAddLink: 0.05,
      probAddNode: 0.01,
      probCancelLink: 0.75,
      maxPerturbation: 50,
      crossOverRate: 0.25,
      staleLimit: 15,
      //coefficients for testing compatibility
      excessCoeff: 1,
      weightDiffCoeff: 0.5,
      compatibilityThreshold: 3
    };


    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Links 2 neurons together - a Link Gene.
     *
     * @class
     */
    class Link{

      #fromNode;
      #toNode;
      #innovID;
      #weight;
      #enabled;

      get from(){ return this.#fromNode }
      get to(){ return this.#toNode }
      get weight(){ return this.#weight }
      get innovID(){ return this.#innovID }
      set weight(w){ this.#weight=w }
      get enabled(){ return this.#enabled }

      turnOff(){ this.#enabled=false; return this; }
      turnOn(){ this.#enabled=true; return this; }
      toggle(n){ this.#enabled=n; return this; }

      /**
       * @param {Node} from
       * @param {Node} to
       * @param {number} wt weight
       * @param {number} innov
       * @param {boolean} on
       */
      constructor(from, to, wt, innov, on=true){
        this.#fromNode = from;
        this.#toNode= to;
        this.#weight=wt;
        this.#innovID=innov;
        this.#enabled = (on !== false);
      }
      /**
       * @return
       */
      mutate(){
        this.#weight= (_.rand() < Params.probMutateWeight) ? _.randMinus1To1() :
                                       _M.clamp(-1, 1, this.#weight + _.randGaussian() / Params.maxPerturbation);
        return this;
      }
      /**
       * @param {Node} from
       * @param {Node} to
       * @return {Link}
       */
      clone(from,to){
        return new Link(from, to, this.#weight, this.#innovID, this.#enabled)
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Innovation is a particular change to a Genome's structure. Each time a
     * genome undergoes a change, that change is recorded as an innovation and
     * is stored in a global historical database.
     * @class
     */
    class Innovation{

      #fromID;
      #toID;
      #history;
      #innovID;

      /**
       * @param {number} from
       * @param {number} to
       * @param {number} innov
       * @param {number[]} history
       */
      constructor(from, to, innov, history){
        this.#fromID = from;
        this.#toID = to;
        this.#innovID= innov;
        _.append(this.#history=[], history,true);
      }
      /**
       * @return {number} size of history
       */
      size(){ return this.#history.length }
      /**
       * @param {Genome} genome
       * @param {Node} from
       * @param {Node} to
       * @return {boolean}
       */
      matches(genome, from, to){
        return genome.size() == this.size() &&
               from.id == this.#fromID && to.id == this.#toID && genome.match(this.#history)
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Used to keep track of all innovations created during the populations
     * evolution, adds all the appropriate innovations.
     *
     * @class
     */
    class InnovHistory{

      #db;

      /**
      */
      constructor(){
        this.#db=[/* Innovation */];
      }
      /**
       * @param {number} i index
       * @return {Innovation}
       */
      getAt(i){
        return this.#db[i]
      }
      /**
       * @param {Function} cb
       * @return {boolean}
       */
      find(cb){
        return this.#db.find(cb)
      }
      /**
       * @param {Innovation} h
       * @return
       */
      append(h){
        this.#db.push(h);
        return this;
      }
    }


    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**A neuron, part of a Genome.
     *
     * @class
     */
    class Node{

      #outputValue;
      #outputLinks;
      #inputSum;
      #layer;
      #type;
      #id;

      get output(){ return this.#outputValue }
      set output(n){ this.#outputValue=n }
      get inputSum(){ return this.#inputSum }
      get id(){ return this.#id }
      get type(){ return this.#type }
      get layer(){ return this.#layer }

      /**Check relative positions of these 2 nodes.
       * @static
       * @param {Node} a
       * @param {Node} b
       * @return {number}
       */
      static checkSibling(a,b){
        return a.layer<b.layer? -1 : (a.layer>b.layer?1:0)
      }

      /**Test if these Nodes are in the same layer.
       * @return {boolean}
       */
      static isSibling(a,b){
        return a.layer==b.layer
      }

      /**
       * @param {number} id
       * @param {NodeType} type
       * @param {number} where layer
       */
      constructor(id, type, where){
        this.#outputLinks= [/*Link*/];
        this.#inputSum = 0;
        this.#outputValue = 0;
        this.#layer = where;
        this.#type=type;
        this.#id = id;
      }
      /**Move node to a different layer.
       * @param {number} layer
       * @return
       */
      moveTo(layer){
        this.#layer=layer;
        return this;
      }
      /**Resets everything.
       * @return
       */
      flush(){
        _.trunc(this.#outputLinks);
        this.#outputValue=0;
        this.#inputSum=0;
        return this;
      }
      /**Change the value of total inputs.
       * @param {number} n
       * @return
       */
      resetInput(n){
        this.#inputSum=n;
        return this;
      }
      /**Add value to the total inputs.
       * @param {number} n
       * @return
       */
      addInput(n){
        this.#inputSum += n;
        return this;
      }
      /**Add a output connection - linking to another node.
       * @param {Link} g
       * @return
       */
      addOutLink(g){
        this.#outputLinks.push(g);
        return this;
      }
      /**Push value downstream to all the output connections.
       * @return
       */
      activate(){
        if(this.#layer != NodeType.INPUT)
          this.#outputValue = this.#sigmoid(this.#inputSum);
        this.#outputLinks.forEach(k=> k.enabled ? k.to.addInput(k.weight * this.#outputValue) : 0);
        return this;
      }
      /**
       * @param {number} x
       * @return {number}
       */
      #sigmoid(x){
        return 1.0 / (1.0 + Math.pow(Math.E, -4.9 * x));
      }
      /**Internal Only.
       * @return {array} output links
       */
      _olinks(){ return this.#outputLinks }
      /**Test if this node is connected to `node`.
       * @param {Node} node
       * @param {boolean}
       */
      isLinked(node){
        let pos= Node.checkSibling(node,this);
        if(pos<0)
          return node._olinks().find(k=> k.to.id == this.id);
        if(pos>0)
          return this.#outputLinks.find(k=> k.to.id == node.id);
        return false;
      }
      /**
       * @return {Node}
       */
      clone() {
        return new Node(this.#id, this.#type, this.#layer);
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**The NeuralNetwork
     *
     * @class
     */
    class Cerveau{

      #outputs;
      #bias;
      #nodes;
      #mesh;
      #inputs;

      /**
       * @param {Genome}
       */
      constructor(g){
        this.#nodes= g.copyNodes();
        this.#mesh=[];
        this.#inputs= this.#nodes.reduce((acc,n)=> n.type==NodeType.INPUT ? acc+1 : acc,0);
        this.#outputs= this.#nodes.reduce((acc,n)=> n.type==NodeType.OUTPUT ? acc+1 : acc,0);
        for(let y=0; y< g.depth; ++y){
          this.#nodes.forEach(n=>{
            if(n.type==NodeType.BIAS){
              this.#bias=n;
            }
            n.layer==y ? this.#mesh.push(n) : 0
          });
        }
        _.assert(this.#bias, `no bias? with depth= ${g.depth}`);
      }
      /**
       * @see update
       * @param {array} data
       * @return {number}
       */
      compute(data){
        return this.update(data)
      }
      /**Do some thinking, given this data.
       *
       * @param {array} data
       * @return {number}
       */
      update(data){
        _.assert(data.length==this.#inputs, `update: expecting ${this.#inputs} inputs but got ${data.length}`);
        this.#nodes.forEach((n,i)=> n.type==NodeType.INPUT ? n.output= data[i] : 0);
        this.#bias.output=Params.BIAS;
        this.#mesh.forEach(n=> n.activate());
        let outs= this.#nodes.reduce((acc,n)=> n.type==NodeType.OUTPUT ? acc.push(n.output)&&acc : acc, []);
        this.#nodes.forEach(n=> n.resetInput(0));
        return outs;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * The whole set of nuclear DNA of an organism.
     * Genetic information of a cell is stored in chemical form in DNA or RNA.
     * The order of the nucleotide bases arranged in the polynucleotide chain determines
     * the genetic instructions. A gene is a sequence stretch of nucleotides which
     * encodes a specific protein. Humans have thousands of genes in their total DNA molecules.
     * The entire nuclear DNA is called the genome of an organism. This DNA is packed into
     * chromosome structures. All gene sequences are called non-repetitive DNA.
     * A genome has many DNA sequences and these are called repetitive DNA.
     * This repetitive DNA also has a function in the gene regulation.
     * The key difference between gene and genome is that a gene is a locus on a
     * DNA molecule whereas genome is a total nuclear DNA.
     *
     * @class
     */
    class Genome{

      #nextNodeID;
      #outputs;
      #inputs;
      #genes;
      #nodes;
      #layers;
      #nnet;
      #fitness;

      get depth(){ return this.#layers }

      /**
       * @param {number} inputNos
       * @param {number} outputNos
       */
      constructor(inputNos, outputNos){
        this.#outputs = outputNos;
        this.#inputs = inputNos;
        this.#init();
      }
      /**
      */
      #init(){
        this.#genes = [];
        this.#nodes = [];
        this.#layers = 0;
        this.#nextNodeID = 1;
        this.#fitness=0;
      }
      /**
      */
      build(){
        this.#init();
        /////[i,i,iiii,o,o,ooo,B]
        for(let i=0; i<this.#inputs; ++i){
          this.#nodes.push(new Node(this.#genNID(), NodeType.INPUT, 0));
        }
        this.#layers += 1;
        for (let i=0; i<this.#outputs; ++i){
          this.#nodes.push(new Node(this.#genNID(), NodeType.OUTPUT,1));
        }
        this.#layers += 1;
        this.#nodes.push(new Node(this.#genNID(), NodeType.BIAS, 0));
        return this;
      }
      /**Get the Link at this index.
       * @param {number} i
       * @return {Link}
       */
      geneAt(i){
        return this.#genes[i];
      }
      /**Get the Node at this index.
       * @param {number} i
       * @return {Node}
       */
      nodeAt(i){
        return this.#nodes[i];
      }
      /**Get the bias Node.
       * @return {Node}
       */
      biasNode(){
        return this.#nodes.find(n=>n.type==NodeType.BIAS)
      }
      /**A shallow copy of all the Nodes.
       * @return {array}
       */
      copyNodes(){
        return this.#nodes.map(n=> n)
      }
      /**True if nodes are not connected.
       * @return {boolean}
       */
      isEmpty(){ return this.#genes.length==0 }
      /**Count the number of Links.
       * @return {number}
       */
      size(){ return this.#genes.length }
      /**Get the fitness score.
       * @return {number}
       */
      getScore(){
        return this.#fitness;
      }
      /**Set the fitness score.
       * @param {number} s
       * @return
       */
      setScore(s){
        this.#fitness=s;
        return this;
      }
      /**Find Node with this id.
       * @param {number} id
       * @return {Node}
       */
      getNode(id){
        return this.#nodes.find(n=> n.id == id);
      }
      /**
      */
      #amalgamate(){
        this.#nodes.forEach(n=> n.flush());
        this.#genes.forEach(g=> g.from.addOutLink(g));
        return this;
      }
      /**Create the corresponding Phenotype.
       * @return
       */
      reify(){
        this.#amalgamate();
        return new Cerveau(this);
      }
      /**Add another Link.
       * @param {InnovHistory} innovDB
       * @return
       */
      addLink(innovDB){
        if(!this.#isPacked()){
          let r1= _.randItem(this.#nodes);
          let r2= _.randItem(this.#nodes);
          let t,inv, _bad= (a,b)=> a.layer == b.layer || a.isLinked(b);

          while(_bad(r1, r2)){
            r1= _.randItem(this.#nodes);
            r2= _.randItem(this.#nodes);
          }
          if(r1.layer > r2.layer){
            t= r2; r2 = r1; r1 = t;
          }
          this.#genes.push(new Link(r1, r2, _.randMinus1To1(), this.getInnov(innovDB, r1, r2)));
          this.#amalgamate();
        }
        return this;
      }
      /**Add another Node.
       * @param {InnovHistory} innovDB
       * @return
       */
      addNode(innovDB){
        if(this.isEmpty()){ return this.addLink(innovDB) }
        let inv,
            newNode,
            bias=this.biasNode(),
            chosen = _.randItem(this.#genes);

        while(chosen.from.id == bias.id && this.size() > 1){
          chosen = _.randItem(this.#genes)
        }
        chosen.turnOff();

        this.#nodes.push(newNode=new Node(this.#genNID(),NodeType.HIDDEN,2));
        this.#genes.push(new Link(chosen.from, newNode, 1,
                                  this.getInnov(innovDB, chosen.from, newNode)));
        this.#genes.push(new Link(newNode, chosen.to, chosen.weight,
                                  this.getInnov(innovDB, newNode, chosen.to)));
        newNode.moveTo(chosen.from.layer+1);

        this.#genes.push(new Link(bias, newNode, 0,
                                  this.getInnov(innovDB, bias, newNode)));

        if(Node.isSibling(newNode, chosen.to)){
          for(let n,i=0; i<this.#nodes.length-1; ++i){ //dont include this newest node
            n=this.#nodes[i];
            if(Node.checkSibling(n,newNode)>=0){
              n.moveTo(n.layer+1);
            }
          }
          ++this.#layers;
        }

        return this.#amalgamate();
      }
      /**
       * @param {InnovHistory} db
       * @param {Node} from
       * @param {Node} to
       * @return {boolean}
       */
      getInnov(db, from, to){
        let newID= Params.nextInnov,
            found= db.find(h=> h.matches(this, from, to));
        if(!found){
          db.append(new Innovation(from.id, to.id, newID, this.#genes.map(g=> g.innovID)));
          ++Params.nextInnov;
        }
        return found ? found.innovID : newID;
      }
      /**
      */
      #isPacked(){
        let totalLinks=0,
            nodesInLayers=[];
        for(let i=0; i<this.#layers; ++i){
          nodesInLayers[i] = 0;
        }
        this.#nodes.forEach(n=> nodesInLayers[ n.layer ] += 1);
        //for each layer the maximum amount of connections is the number in this layer * the number of this.nodes infront of it
        //so lets add the max for each layer together and then we will get the maximum amount of connections in the network
        for(let ns, i=0; i<this.#layers-1; ++i){
          ns = 0;
          for(let j= i+1; j< this.#layers; ++j){ //for each layer infront of this layer
            ns += nodesInLayers[j];
          }
          totalLinks += nodesInLayers[i] * ns;
        }
        return totalLinks <= this.size();
      }
      /**Maybe make some random changes to itself.
       * @param {InnovHistory} innovDB
       * @return
       */
      mutate(innovDB){
        if(this.isEmpty()){
          this.addLink(innovDB);
        }
        if(_.rand < Params.probMutateLink){
          this.#genes.forEach(g=> g.mutate());
        }
        if(_.rand() < Params.probAddLink){
          this.addLink(innovDB);
        }
        if(_.rand() < Params.probAddNode){
          this.addNode(innovDB);
        }
        return this;
      }
      /**Internal Only
       * @return
       */
      _introspect(){
        let y=0, nid=0;
        this.#nodes.forEach(n=>{
          if(n.layer>y){ y = n.layer }
          if(n.id>nid){ nid=n.id }
        });
        this.#nextNodeID=nid+1;
        this.#layers=y+1;
        return this;
      }
      /**Clone the incoming Node.
       * @param {Node} n
       * @return
       */
      cloneNode(n){
        this.#nodes.push(n.clone());
        return this;
      }
      /**Clone the incoming Link.
       * @param {Link} g
       * @param {boolean} en enable
       * @return
       */
      cloneGene(g,en){
        this.#genes.push(g.clone(this.getNode(g.from.id), this.getNode(g.to.id)).toggle(en));
        return this;
      }
      /**Mate with this other Node.
       * @param {Node} other
       * @return {Node}
       */
      crossOver(other){
        let isEnabled = [],
            newGenes = [],
            child = new Genome(this.#inputs, this.#outputs);
        this.#genes.forEach(g=>{
          let en = true;
          let p2 = other.findGene(g.innovID);
          if(p2){
            if(!g.enabled || !p2.enabled){
              if(_.rand() < Params.probCancelLink){
                en = false;
              }
            }
            newGenes.push(_.randAorB(g,p2));
          }else{
            //disjoint or excess gene
            en = g.enabled;
            newGenes.push(g);
          }
          isEnabled.push(en);
        });
        //since all excess and disjovar genes are inherrited from the more fit parent (this Genome) the childs structure is no different from this parent | with exception of dormant connections being enabled but this wont effect this.nodes
        //so all the this.nodes can be inherrited from this parent
        this.#nodes.forEach(n=> child.cloneNode(n));
        //clone all the connections so that they connect the childs new this.nodes
        newGenes.forEach((g,i)=> child.cloneGene(g, isEnabled[i]));
        return child._introspect().#amalgamate();
      }
      /**Look for a Link with this innovation.
       * @param {number} innov
       * @return {Link}
       */
      findGene(innov){
        return this.#genes.find(g=> g.innovID == innov);
      }
      /**
       * @return {Genome}
       */
      clone(){
        let rc= new Genome(this.#inputs, this.#outputs);
        this.#nodes.forEach(n=> rc.cloneNode(n));
        this.#genes.forEach(g=> rc.cloneGene(g, g.enabled));
        return rc._introspect().#amalgamate();
      }
      /**Check if all the Links are recorded already in the history.
       * @param {number[]} history
       * @return {boolean}
       */
      match(history){
        return this.#genes.every(g=> history.includes(g.innovID))
      }
      /**Count all matching Links
       * @param {Genome} g2
       * @return {number}
       */
      countMatching(g2){
        let c=0;
        this.#genes.forEach(g=> g2.findGene(g.innovID) ? ++c : 0);
        return c;
      }
      /**
      */
      #genNID(){ return this.#nextNodeID++ }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Species
     * @class
     */
    class Species{

      #bestScore;
      #avgScore;
      #members;
      #alpha;
      #stale;

      get avgScore(){ return this.#avgScore }
      get alpha(){ return this.#alpha }
      get bestScore(){ return this.#bestScore }

      /**
       * @param {Genome}
       */
      constructor(g){
        this.#bestScore = g.getScore();
        this.#alpha = g.clone();
        this.#members = [];
        this.#avgScore=0;
        this.#stale= 0;
        this.#members.push(g);
      }
      /**Check `g`'s compatibility.'
       * @param {Genome} g
       * @return {boolean}
       */
      compatible(g){
        let gdiff = this.#getNotMatched(g, this.#alpha);
        let wdiff = this.#avgWeightDiff(g, this.#alpha);
        let normaliser = g.size() - 20;
        if(normaliser < 1){ normaliser = 1 }
        //compatibility formula
        return Params.compatibilityThreshold > (Params.excessCoeff * gdiff /normaliser + Params.weightDiffCoeff * wdiff);
      }
      /**Include this Genome into this specie.
       * @param {Genome} g
       * @return
       */
      add(g){
        if(0 && g.getScore()>this.#bestScore){
          this.#bestScore=g.getScore();
          this.#alpha=g;
        }
        this.#members.push(g);
      }
      /**
       * @param {Genome} g1
       * @param {Genome} g2
       * @return {number}
       */
      #getNotMatched(g1, g2){
        return g1.size() + g2.size() - 2 * g1.countMatching(g2);
      }
      /**
       * @param {Genome} g1
       * @param {Genome} g2
       * @return {number}
       */
      #avgWeightDiff(g1,g2){
        let diff= 0, matched=0;
        for(let i= 0; i < g1.size(); ++i){
          for(let j= 0; j < g2.size(); ++j){
            if(g1.geneAt(i).innovID == g2.geneAt(j).innovID){
              ++matched;
              diff += Math.abs(g1.geneAt(i).weight - g2.geneAt(j).weight);
              break;
            }
          }
        }
        return matched==0 ? (g1.isEmpty()||g2.isEmpty()?0:100) : diff/matched;
      }
      /**Sort everything with `best` stuff in the front.
       * @return
       */
      sort(){
        this.#members.sort(_.comparator(_.SORT_DESC,a=>a.getScore(),b=>b.getScore()));
        if(this.#members.length == 0){
          this.#stale= 200;
        }else if(this.#members[0].getScore() > this.#bestScore){
          this.#bestScore= this.#members[0].getScore();
          this.#alpha= this.#members[0].clone();
          this.#stale = 0;
        }else{
          ++this.stale;
        }
        return this;
      }
      /**
       * @return {boolean}
       */
      isEmpty(){ return this.#members.length==0 }
      /**
       * @return {number}
       */
      size(){ return this.#members.length }
      /**
       * @return
       */
      setAverage(){
        this.#avgScore = this.#members.reduce((acc,g)=> acc + g.getScore(),0) / this.size();
        return this;
      }
      /**
       * @param {InnovHistory} innovDB
       * @return {Genome}
       */
      spawn(innovDB){
        let baby;
        if(_.rand() < Params.crossOverRate){
          baby = this.#randAny().clone();
        }else{
          let p1 = this.#randAny();
          let p2 = this.#randAny();
          if(p1.getScore() < p2.getScore()){
            baby = p2.crossOver(p1);
          }else{
            baby = p1.crossOver(p2);
          }
        }
        return baby.mutate(innovDB);
      }
      /**
       * @return {Genome}
       */
      #randAny(){
        let slice = this.#members.reduce((acc,g)=> acc+g.getScore(),0) * _.rand();
        let sum = 0;
        let rc= this.#members.find(g=>{
          sum += g.getScore();
          if(sum > slice){ return true }
        });
        return rc || this.#members[0];
      }
      /**Get rid of bottom half of this specie.
       * @return
       */
      cull(){
        if(this.#members.length > 2){
          for(let i= this.#members.length / 2; i < this.#members.length; ++i){
            this.#members.splice(i, 1);
            --i;
          }
        }
        return this;
      }
      /**Remove all members.
       * @return
       */
      flush(){
        _.trunc(this.#members);
        return this;
      }
      /**Normalize all scores.
       * @return
       */
      normalize(){
        this.#members.forEach(g=> g.setScore(g.getScore()/ this.size()));
        this.#avgScore = this.#members.reduce((acc,g)=> acc + g.getScore(),0) / this.size();
        return this;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**NeatGA
     * @class
     */
    class NeatGA{

      #numOutputs;
      #numInputs;

      #db;
      #genomes;
      #species;
      #size;
      #gen;

      get popSize(){ return this.#size }

      /**
       * @param {number} size
       * @param {number} inputNo
       * @param {number} outputNo
       */
      constructor(size, inputNo, outputNo){
        this.#numOutputs=outputNo;
        this.#numInputs=inputNo;
        this.#size=size;
        this.#db = new InnovHistory();
        this.#gen = 1;
        this.#genomes=[];
        this.#species = [];
        for(let i=0; i<size; ++i){
          this.#genomes= _.fill(size, ()=> new Genome(inputNo,outputNo).build().mutate(this.#db))
        }
      }
      /**Get the current generation count.
       * @return {number}
       */
      curGen(){
        return this.#gen;
      }
      /**Reify all genomes into their Phenotypes.
       * @return {array}
       */
      createPhenotypes(){
        return this.#genomes.map(g=> g.reify());
      }
      /**Create a new generation by studying the results of the previous generation.
       * @param {array} scores
       * @return {array} new set of Phenotypes.
       */
      epoch(scores){
        let prevBest= this.#gen==1 ? null : this.#genomes[0];

        this.#speciate();
        this.#calcFitness(scores);
        this.#sortSpecies();
        this.#cullSpecies();
        this.#killStaleSpecies();
        this.#killBadSpecies();

        let children = [],
            sum = this.#sumAvgScore();
        this.#species.forEach(s=>{
          children.push(s.alpha.clone());
          let count = int(s.avgScore / sum * this.popSize) - 1;
          //the number of children this this.species is allowed, note -1 is because the champ is already added
          for(let i=0; i < count; ++i){
            children.push(s.spawn(this.#db));
          }
        });
        if(children.length < this.popSize){
          children.push(this.#species[0].alpha.clone());
        }
        while(children.length < this.popSize){
          children.push(this.#species[0].spawn(this.#db));
        }
        _.append(this.#genomes, children, true);
        this.#gen += 1;
        return this.createPhenotypes();
      }
      /**
      */
      #speciate(){
        this.#species.forEach(s=> s.flush());
        this.#genomes.forEach((g,i)=>{
          i=this.#species.find(s=> s.compatible(g));
          i ? i.add(g) : this.#species.push(new Species(g));
        });
      }
      /**
      */
      #calcFitness(scores){
        this.#genomes.forEach((g,i)=> g.setScore(scores[i]));
        return this;
      }
      /**
      */
      #sortSpecies(){
        this.#species.forEach(s=> s.sort());
        this.#species.sort(_.comparator(_.SORT_DESC,a=>a.bestScore, b=>b.bestScore));
      }
      /**
      */
      #killStaleSpecies(){
        for(let i=2; i < this.#species.length; ++i){
          if(this.#species[i].stale >= Params.staleLimit){
            this.#species.splice(i, 1);
            --i;
          }
        }
      }
      /**
      */
      #killBadSpecies(){
        let size=this.popSize,
            sum= this.#sumAvgScore();
        for(let i=1; i < this.#species.length; ++i){
          if(this.#species[i].avgScore / sum * size < 1){
            this.#species.splice(i, 1);
            --i;
          }
        }
      }
      /**
      */
      #sumAvgScore(){
        return this.#species.reduce((acc,s)=> acc + s.avgScore,0)
      }
      /**
      */
      #cullSpecies(){
        this.#species.forEach(s=> s.cull().normalize());
        return this;
      }
    }


    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      NeatGA, Genome, Link, Node, Species,
      Innovation,InnovHistory,
      configParams(options){ return _.inject(Params,options) }
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"))
  }else{
    gscope["io/czlab/mcfud/algo/NEAT_CBullet"]=_module
  }

})(this)




/* Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright © 2025, Kenneth Leung. All rights reserved. */

;(function(gscope,UNDEF){

  "use strict";

  /**Create the module.
   */
  function _module(Mcfud){

    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();
    const _M= Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();
    const int=Math.floor;
    const {u:_, is}= Core;

    /**
     * @module mcfud/algo/NEAT_Buckland
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @typedef {object} NeuronType
     * @property {number} INPUT
     * @property {number} HIDDEN
     * @property {number} OUTPUT
     * @property {number} BIAS
     * @property {number} NONE
     */
    const NeuronType={ INPUT: 1, BIAS: 2, HIDDEN: 3, OUTPUT: 4, NONE: 911 };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @typedef {object} InnovType
     * @property {number} NEURON
     * @property {number} LINK
     */
    const InnovType={ NEURON:0, LINK:1 }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Select one of these types when updating the network if snapshot is chosen
     * the network depth is used to completely flush the inputs through the network.
     * active just updates the network each timestep.
     * @typedef {object} RunType
     * @property {number} SNAPSHOT
     * @property {number} ACTIVE
     */
    const RunType={ SNAPSHOT:7770, ACTIVE:8881 }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const Params={
      numInputs: 0,
      numOutputs: 0,
      BIAS: 1,//-1,
      //starting value for the sigmoid response
      sigmoidResponse:1,
      //number of times we try to find 2 unlinked nodes when adding a link.
      addLinkAttempts:5,
      //number of attempts made to choose a node that is not an input
      //node and that does not already have a recurrently looped connection to itself
      findLoopedLink: 5,
      //the number of attempts made to find an old link to prevent chaining in addNeuron
      findOldLink: 5,
      //the chance, each epoch, that a neuron or link will be added to the genome
      chanceAddLink:0.07,
      chanceAddNode:0.03,
      chanceRecurrent: -1,//0.05,
      //mutation probabilities for mutating the weights
      mutationRate:0.8,
      maxWeightJiggle:0.5,
      chanceSetWeight:0.1,
      //probabilities for mutating the activation response
      activationMutation:0.1,
      maxActivationJiggle:0.1,
      //the smaller the number the more species will be created
      compatThreshold:0.26,
      //during fitness adjustment this is how much the fitnesses of
      //young species are boosted (eg 1.2 is a 20% boost)
      youngFitnessBonus:1.3,
      //if the species are below this age their fitnesses are boosted
      youngBonusAge:10,
      //number of population to survive each epoch. (0.2 = 20%)
      survivalRate:0,
      //if the species is above this age their fitness gets penalized
      oldAgeThreshold:50,
      //by this much
      oldAgePenalty:0.7,
      crossOverRate:0.7,
      //how long we allow a species to exist without any improvement
      noImprovements:15,
      //maximum number of neurons permitted in the network
      maxNNetNeurons:100,
      numBestElites:4,
      fitFunc: function(seed=0){ return new FitFunc(seed) },
      sigmoid: function(netinput, response){
        return 1 / ( 1 + Math.exp(-netinput / response))
      },
      sigmoid2: function(x){
        //y=1/(1+e^(-x))
        return 1 / (1 + Math.pow(Math.E, -4.9*x))
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    function _isOUTPUT(n){ return n.neuronType == NeuronType.OUTPUT }
    function _isBIAS(n){ return n.neuronType == NeuronType.BIAS }
    function _isINPUT(n,bias=false){
      return n.neuronType == NeuronType.INPUT || (bias && n.neuronType==NeuronType.BIAS);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**A numeric fitness object.
     * @class
     */
    class FitFunc{
      #value;
      /**
       * @param {any} seed
      */
      constructor(seed){
        this.#value=seed;
      }
      /**Update the fitness with this value.
       * @param {any} v
       */
      update(v){
        this.#value=v;
        return this;
      }
      /**Get the score.
       * @return {any}
       */
      score(){ return this.#value }
      /**Create a clone of this fitness.
       * @return {FitFunc}
       */
      clone(){
        return new FitFunc(this.#value)
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @class
     */
    class Coord{

      #x;
      #y;

      get x(){ return this.#x }
      get y(){ return this.#y }

      /**
       * @param {number} x
       * @param {number} y
       */
      constructor(x=0,y=0){
        this.#x=x;
        this.#y=y;
      }
      /**
       * @return {Coord}
       */
      clone(){
        return new Coord(this.#x, this.#y)
      }
      /**
       * @param {number} x
       * @param {number} y
       * @return {Coord}
       */
      static dft(){
        return new Coord(0,0)
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    function _strNodeType(t){
      switch(t){
        case NeuronType.INPUT: return "input";
        case NeuronType.BIAS: return "bias";
        case NeuronType.HIDDEN: return "hidden";
        case NeuronType.OUTPUT: return "output";
      }
      _.assert(false, `bad node type ${t}`);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**NeuronGene, part of a Genome.
     * @class
     */
    class NeuronGene{

      #activation;
      #recurrent;
      #type;
      #pos;
      #id;

      /**
      */
      get activation(){ return this.#activation }
      get neuronType(){ return this.#type }
      get recur(){ return this.#recurrent }
      get id(){ return this.#id }
      get posY(){ return this.#pos.y }
      get posX(){ return this.#pos.x }

      /**
      */
      set activation(a){ this.#activation=a }

      /**
       * @param {number} id
       * @param {NeuronType} type
       * @param {Coord} pos
       * @param {boolean} recur
       */
      constructor(id, type, pos=null,recur=false){
        _.assert(id>0, `creating a neuron with a bad id ${id}`);
        this.#pos= pos ? pos.clone() : Coord.dft();
        this.#recurrent= (recur===true);
        this.#activation=1;
        this.#id=id;
        this.#type= type;
      }
      /**
       * @return {String}
      */
      prn(){
        return `${_strNodeType(this.#type)}#[${this.#id}]`
      }
      /**
      */
      setRecur(r){
        this.#recurrent=r;
        return this;
      }
      /**
       * @return {NeuronGene} copy
       */
      clone(){
        const rc= new NeuronGene(this.#id,this.#type,
                                 this.#pos, this.#recurrent);
        rc.activation= this.#activation;
        return rc;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**LinkGene, part of a Genome, links NeuronGene A to NeuronGene B.
     * @class
     */
    class LinkGene{

      #fromNeuron;
      #toNeuron;
      #recurrent;
      #enabled;
      #weight;

      /**
      */
      get fromNeuron(){ return this.#fromNeuron }
      get toNeuron(){ return this.#toNeuron }
      get enabled(){ return this.#enabled }
      get weight(){ return this.#weight }
      get recur(){ return this.#recurrent }

      /**
      */
      set weight(w){ this.#weight=w }

      /**
       * @param {number} from
       * @param {number} to
       * @param {boolean} enable
       * @param {number} w
       * @param {boolean} rec
       */
      constructor(from, to, enable=true, w=null, recur=false){
        this.#fromNeuron= from;
        this.#toNeuron= to;
        this.#recurrent=(recur===true);
        this.#enabled=(enable !== false);
        this.#weight= w===null||isNaN(w) ? _.randMinus1To1() : w;
      }
      /**
       * @return {LinkGene} A copy
      */
      clone(){
        return new LinkGene(this.#fromNeuron,this.#toNeuron,
                            this.#enabled, this.#weight, this.#recurrent)
      }
      /**
       * @param {boolean} r
       * @return {boolean} r
       */
      setRecur(r){
        this.#recurrent=r;
        return this;
      }
      /**
       * @param {boolean} e
       * @return {boolean} e
       */
      setEnabled(e){
        this.#enabled=e;
        return this;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Innovation is a particular change to a Genome's structure. Each time a
     * genome undergoes a change, that change is recorded as an innovation and
     * is stored in a global historical database.
     * @class
     */
    class Innov{

      #innovationType;
      #innovationID;
      #neuronID;
      #pos;
      #neuronType;
      #neuronIn;
      #neuronOut;

      /**
      */
      get innovType(){ return this.#innovationType }
      get neuronID(){ return this.#neuronID }
      get IID(){ return this.#innovationID }
      get pos() { return this.#pos }
      get neuronIn(){ return this.#neuronIn }
      get neuronOut(){ return this.#neuronOut }
      get neuronType(){ return this.#neuronType }

      /**
       * @param {InnovHistory} db
       * @param {number} from
       * @param {number} to
       * @param {InnovType} type
       * @param {array} extra [id,NeuronType]
       * @param {Coord} pos
       */
      constructor(db, from, to, type, extra=null, pos=null){
        this.#pos= pos ? pos.clone() : Coord.dft();
        this.#innovationID= db.nextIID();
        this.#innovationType=type;
        this.#neuronIn= from;
        this.#neuronOut= to;
        if(is.vecN(extra,2,true)){
          this.#neuronID= extra[0];
          this.#neuronType= extra[1];
        }else{
          this.#neuronID= -31;
          this.#neuronType= NeuronType.NONE;
        }
        db.add(this);
      }
      /**
       * @param {InnovHistory} db
       * @param {number} nid neuron id
       * @param {NeuronType} type
       * @param {Coord} pos
       * @return {Innov}
      */
      static from(db, nid, type, pos){
        return new Innov(db, -71,-99, InnovType.NEURON, [nid, type], pos)
      }
    }

    /**
    */
    function _calcSplits(inputs,outputs){
      return [ 1/(inputs+2), 1/(outputs+1) ]
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Used to keep track of all innovations created during the populations
     * evolution, adds all the appropriate innovations.
     * @class
     */
    class InnovHistory{

      #NEURON_COUNTER;
      #INNOV_COUNTER;
      #inputs;
      #outputs;
      #vecNodes;
      #vecInnovs;

      /**Initialize the history database using these current
       * inputs and outputs.
       * @param {number} inputs
       * @param {number} outputs
       */
      constructor(inputs, outputs){

        let [iXGap, oXGap] = _calcSplits(inputs,outputs);
        let nObj, nid=0;

        this.#NEURON_COUNTER=0;
        this.#INNOV_COUNTER=0;
        this.#outputs=outputs;
        this.#inputs=inputs;
        this.#vecInnovs=[];
        this.#vecNodes=[];

        for(let i=0; i<inputs; ++i){
          nObj={t: NeuronType.INPUT, id: ++nid,  co: new Coord((i+2)*iXGap,0)};
          this.#vecNodes.push(nObj);
          Innov.from(this, nObj.id, nObj.t,  nObj.co);
        }

        nObj= {t:NeuronType.BIAS, id: ++nid, co: new Coord(iXGap,0)};
        this.#vecNodes.push(nObj);
        Innov.from(this,nObj.id, nObj.t, nObj.co);

        for(let i=0; i<outputs; ++i){
          nObj={t:NeuronType.OUTPUT, id: ++nid, co: new Coord((i+1)*oXGap,1) };
          this.#vecNodes.push(nObj);
          Innov.from(this, nObj.id, nObj.t, nObj.co);
        }

        _.assert(nid==inputs+outputs+1,"bad history db ctor - mismatched neuron ids");
        _.assert(nid==this.#vecNodes.at(-1).id, "bad history db ctor - erroneous last neuron id");
        this.#NEURON_COUNTER= nid;

        //_.log(`InnovHistory: inputs=${inputs}, outputs=${outputs}, last neuron id=> ${nid}`);

        //connect each input & bias neuron to each output neuron
        outputs= this.#vecNodes.filter(n=>n.t == NeuronType.OUTPUT);
        inputs= this.#vecNodes.filter(n=>n.t != NeuronType.OUTPUT);
        inputs.forEach(i=> outputs.forEach(o => new Innov(this, i.id, o.id, InnovType.LINK)));
      }
      /**Create a new genome.
       * @param {array} nodes output
       * @param {array} links output
       */
      sample(nodes,links){
        //make neuron genes then connect each input & bias neuron to each output neuron
        let ins= this.#vecNodes.filter(nObj=>nObj.t != NeuronType.OUTPUT);
        let os= this.#vecNodes.filter(nObj=>nObj.t == NeuronType.OUTPUT);
        this.#vecNodes.forEach(nObj=> nodes.push(new NeuronGene(nObj.id, nObj.t, nObj.co)));
        ins.forEach(i=> os.forEach(o=> links.push(new LinkGene(i.id, o.id))));
        return this;
      }
      /**
       * @return {number} next neuron id
      */
      #genNID(){ return ++this.#NEURON_COUNTER }
      /**
       * @return {number} next innovation number
       */
      nextIID(){ return ++this.#INNOV_COUNTER }
      /**Checks to see if this innovation has already occurred. If it has it
       * returns the innovation ID. If not it returns a negative value.
       * @param {number} from
       * @param {number} out
       * @param {InnovType} type
       * @return {number}
       */
      check(from, out, type){
        _.assert(from>0 && out>0, `checking innov with bad neuron ids: from: ${from}, to: ${out}`);
        const rc= this.#vecInnovs.find(cur=> cur.innovType == type &&
                                             cur.neuronIn == from && cur.neuronOut == out);
        return rc ? rc.IID : -51;
      }
      /**
       * @param {Innov} n
       */
      add(n){
        this.#vecInnovs.push(n)
      }
      /**Creates a new innovation.
       * @param {number} from
       * @param {number} to
       * @param {InnovType} innovType
       * @param {NeuronType} neuronType
       * @param {Coord} pos
       * @return {Innov}
       */
      create(from, to, innovType, neuronType=NeuronType.NONE, pos=null){
        let i;
        if(innovType==InnovType.NEURON){
          _.assert(neuronType != NeuronType.NONE, "create-innov: unexpected bad neuron type");
          _.assert(from>0&&to>0, `create-innov: bad neuron ids: from: ${from} to: ${to}`);
          i= new Innov(this, from, to, innovType, [this.#genNID(),neuronType], pos)
        }else{
          i= new Innov(this, from, to, innovType, null, pos);
        }
        return i;
      }
      /**
       * @param {number} nid Neuron Id.
       * @return {NeuronGene}
       */
      createNeuronFromID(nid){
        const i= this.#vecInnovs.find(n=> n.neuronID==nid);
        return i ? (new NeuronGene(nid, i.neuronType, i.pos))
                 : _.assert(false, "unknown neuron id not found in history.");
      }
      /**
       * @param {number} inv .index into the list of innovations.
       * @return {number} Neuron ID or -1
      */
      getNeuronID(inv){
        const rc= this.#vecInnovs.find(n=> n.IID == inv);
        return rc ? rc.neuronID : -41;
      }
      /**
       * @param {LinkGene} gene
       * @param {InnovType} type
       * @return {number} innov id.
       */
      getIID(gene, type=InnovType.LINK){
        return this.check(gene.fromNeuron, gene.toNeuron, type)
      }
      /**
       * @param {LinkGene} gene
       * @param {InnovType} type
       * @return {Innov} innov
       */
      getInnov(gene, type=InnovType.LINK){
        return this.#vecInnovs.find(i=> i.innovType == type &&
                                        i.neuronIn == gene.fromNeuron && i.neuronOut == gene.toNeuron)
      }
      /**
       * @return {number}
       */
      countInputs(){ return this.#inputs }
      /**
       * @return {number}
       */
      countOutputs(){ return this.#outputs }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**NLink
     * @class
     */
    class NLink{

      #weight;
      #from;
      #out;
      #recurrent;

      /**
      */
      get weight(){ return this.#weight }
      get from(){ return this.#from }

      /**
       * @param {number} w
       * @param {NNeuron} from
       * @param {NNeuron} out
       * @param {boolean} recur
       */
      constructor(w, from, out, recur=false){
        this.#weight=w;
        this.#from=from;
        this.#out=out;
        this.#recurrent= (recur===true);
      }
      /**
       * @return {NLink} A copy
       */
      clone(){
        return new NLink(this.#weight, this.#from, this.#out, this.#recurrent)
      }
      /**
       * Create a link between these two neurons and
       * assign the weight stored in the gene.
       * @param {LinkGene} gene
       * @param {NNeuron} from
       * @param {NNeuron} to
       * @return {NLink}
       */
      static from(gene,from,to){
        const rc= new NLink(gene.weight, from, to, gene.recur);
        //add new links to neurons
        from.addLinkOut(rc);
        to.addLinkIn(rc);
        return rc;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**NNeuron
     * @class
     */
    class NNeuron{

      #vecLinksOut;
      #activation;
      #vecLinksIn;
      #neuronType;
      #neuronID;
      #output;
      #pos;

      /**
      */
      get activation() { return this.#activation }
      get neuronType() { return this.#neuronType }
      get id() { return this.#neuronID }
      get output() { return this.#output }

      /**
      */
      set output(o) { this.#output=o }


      /**
       * @param {number} id
       * @param {NeuronType} type
       * @param {Coord} pos
       * @param {number} act_response
       */
      constructor(id,type, pos, act_response){
        this.#pos= pos ? pos.clone() : Coord.dft();
        this.#activation=act_response;
        this.#neuronType=type;
        this.#neuronID=id;
        this.#output=0;
        this.#vecLinksIn=[];
        this.#vecLinksOut=[];
      }
      /**
      */
      _cpy(output,inLinks,outLinks){
        this.#vecLinksOut=outLinks.map(v=> v.clone());
        this.#vecLinksIn=inLinks.map(v=> v.clone());
        this.#output=output;
        return this;
      }
      /**
       * @return {String}
       */
      prn(){
        return `node(${_strNodeType(this.#neuronType)})#[${this.#neuronID}]`;
      }
      /**
      */
      flush(){
        this.#output=0;
        return this;
      }
      /**
       * @return {NNeuron} A copy
      */
      clone(){
        return new NNeuron(this.#neuronID,this.#neuronType,this.#pos,this.#activation).
               _cpy(this.#output,this.#vecLinksIn, this.#vecLinksOut)
      }
      /**
       * @param {Function} func
       * @return {any}
       */
      funcOverInLinks(func){
        return func(this.#vecLinksIn)
      }
      /**
       * @param {NLink} n
       */
      addLinkIn(n){
        this.#vecLinksIn.push(n);
        return this;
      }
      /**
       * @param {NLink} o
       * @return {NNeuron} this
       */
      addLinkOut(o){
        this.#vecLinksOut.push(o);
        return this;
      }
      /**
       * @param {NeuronGene} n
       * @return {NNeuron}
       */
      static from(n){
        return new NNeuron(n.id, n.neuronType, n.pos, n.activation)
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**NeuralNetwork
     * @class
     */
    class NeuralNet{

      #vecNeurons;
      #depth;

      /**
       * @param {NNeuron[]} neurons
       * @param {number} depth
       */
      constructor(neurons, depth){
        _.append(this.#vecNeurons=[], neurons, true);
        this.#depth=depth;
        neurons.length=0;
      }
      /**
       * @return {NeuralNet}
       */
      clone(){
        return new NeuralNet(this.#vecNeurons.map(n=>n.clone()),this.#depth)
      }
      /**Update network for this clock cycle.
       * @param {number[]} inputs
       * @param {RunType} type
       * @return {number[]} outputs
       */
      compute(inputs,type=RunType.ACTIVE){
        return this.update(inputs, type)
      }
      /**Update network for this clock cycle.
       * @param {number[]} inputs
       * @param {RunType} type
       * @return {number[]} outputs
       */
      update(inputs, type=RunType.ACTIVE){
        //if the mode is snapshot then we require all the neurons to be
        //iterated through as many times as the network is deep. If the
        //mode is set to active the method can return an output after just one iteration
        let n,jobDone,outputs=[],
            loopCnt= type==RunType.SNAPSHOT ? this.#depth : 1;
        function _sum(a){
          return a.reduce((acc,k)=> acc + k.weight * k.from.output,0);
        }
        while(loopCnt--){
          outputs.length=0;
          jobDone=0;//expect 2 jobs (bind input values & bias)
          n=0;
          for(let obj,j=0; j < this.#vecNeurons.length; ++j){
            obj=this.#vecNeurons[j];
            if(_isINPUT(obj)){
              _.assert(n<inputs.length, `NeuralNet: update with mismatched input size ${inputs.length}`);
              obj.output=inputs[n++];
              if(n==inputs.length){++jobDone}
            }else if(_isBIAS(obj)){
              obj.output=Params.BIAS;
              ++jobDone;
            }
            if(jobDone==2){break}
          }
          //now deal with the other neurons...
          this.#vecNeurons.forEach(obj=>{
            if(!_isINPUT(obj,true)){
              obj.output = Params.sigmoid(obj.funcOverInLinks(_sum), obj.activation);
              _isOUTPUT(obj) ? outputs.push(obj.output) : 0;
            }
          });
        }

        if(type == RunType.SNAPSHOT){
          this.#vecNeurons.forEach(n=> n.flush());
        }

        /////
        return outputs;
      }
    }

    /**
     * @param {NeuronGene} from
     * @param {NeuronGene} to
     * @return {Coord}
     */
    function _splitBetween(from,to){
      _.assert(from && to, `splitBetween: unexpected null params: from: ${from}, to: ${to}`);
      return new Coord((from.posX + to.posX)/2, (from.posY + to.posY)/2)
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * The whole set of nuclear DNA of an organism.
     * Genetic information of a cell is stored in chemical form in DNA or RNA.
     * The order of the nucleotide bases arranged in the polynucleotide chain determines
     * the genetic instructions. A gene is a sequence stretch of nucleotides which
     * encodes a specific protein. Humans have thousands of genes in their total DNA molecules.
     * The entire nuclear DNA is called the genome of an organism. This DNA is packed into
     * chromosome structures. All gene sequences are called non-repetitive DNA.
     * A genome has many DNA sequences and these are called repetitive DNA.
     * This repetitive DNA also has a function in the gene regulation.
     * The key difference between gene and genome is that a gene is a locus on a
     * DNA molecule whereas genome is a total nuclear DNA.
     *
     * @class
     */
    class Genome{

      #vecNeurons;
      #vecLinks;
      #non_ins;
      #ins;
      #db;

      #genomeID;
      #fitness;

      //its fitness score after it has been placed into a species and adjusted accordingly
      #adjScore;

      //the number of offspring is required to spawn for the next generation
      #spawnCnt;
      #inputs;
      #outputs;

      //keeps a track of which species this genome is in
      #species;

      /**
      */
      get spawnCnt() { return this.#spawnCnt }
      get adjScore(){ return this.#adjScore }
      get id(){ return this.#genomeID }

      /**
      */
      set spawnCnt(n) { this.#spawnCnt = n }

      /**A genome basically consists of a vector of link genes,
       * a vector of neuron genes and a fitness score.
       * @param {InnovHistory} db
       * @param {number} gid
       */
      constructor(db,gid){
        this.#fitness=Params.fitFunc(0);
        this.#vecNeurons=[];
        this.#vecLinks=[];
        this.#genomeID= gid;
        this.#species=0;
        this.#db=db;
        //adjusted score via species
        this.#adjScore=0;
        this.#spawnCnt=0;
        this.#inputs=db.countInputs();
        this.#outputs=db.countOutputs();
        if(gid>0)
          db.sample(this.#vecNeurons, this.#vecLinks);
        this.#segregate();
      }
      /**
      */
      #segregate(){
        this.#dbgCtor();
        this.#non_ins=[];
        this.#ins=[];
        this.#sortGenes();
        this.#vecNeurons.forEach(n=> _isINPUT(n,true) ? this.#ins.push(n) : this.#non_ins.push(n));
      }
      /**
       * @return {String}
       */
      #prnNodes(){
        return this.#vecNeurons.reduce((acc,n,i)=> acc += ((i==0)?"":", ") + n.prn(), "")
      }
      /**
      */
      #dbgCtor(){
        if(0 && this.#genomeID>0)
          _.log(`genome(${this.#genomeID}):${this.#prnNodes()}`);
      }
      /**
       * @return {String}
       */
      dbgState(){
        return `{nodes=${this.#prnNodes()},links=${this.#vecLinks.length}}`
      }
      /**
       * @param {number} n
       */
      adjustScore(n){
        this.#adjScore=n;
        return this;
      }
      /**
       * @return {number} number of neurons
       */
      numNeurons() { return this.#vecNeurons.length }
      /**
       * @return {number} number of links
      */
      numGenes() { return this.#vecLinks.length }
      /**
       * @param {any} num fitness value
       */
      setScore(num){
        this.#fitness.update(num);
        return this;
      }
      /**
       * @return {number}
       */
      getScore(){ return this.#fitness.score() }
      /**
       * @return {LinkGene}
       */
      geneAt(i) { return this.#vecLinks[i]  }
      /**
       * @return {NeuronGene}
       */
      neuronAt(i) { return this.#vecNeurons[i] }
      /**
       * @param {number} newid
       */
      mutateGID(newid){
        _.assert(newid>0, "bad genome id, must be positive");
        this.#genomeID=newid;
        return this;
      }
      /**
      */
      _specialClone(id, nodes,links){
        this.#non_ins=[];
        this.#ins=[];
        nodes.forEach(n=> _isINPUT(n,true) ? this.#ins.push(n) : this.#non_ins.push(n));
        _.append(this.#vecNeurons, nodes, true);
        _.append(this.#vecLinks, links, true);
        this.#genomeID=id;
        nodes.length=0;
        links.length=0;
        return this;
      }
      /**Create a neural network from the genome.
       * @param {number} depth
       * @return {NeuralNet} newly created ANN
       */
      createPhenotype(depth){
        const vs= this.#vecNeurons.map(g=> NNeuron.from(g));
        this.#vecLinks.forEach(k=>
          k.enabled? NLink.from(k, vs.find(n=> n.id== k.fromNeuron),
                                   vs.find(n=> n.id== k.toNeuron)) :0);
        return new NeuralNet(vs, depth);
      }
      /**
       * @return {InnovHistory}
       */
      history(){ return this.#db }
      /**
      */
      #randAny(){ return _.randItem(this.#vecNeurons) }
      /**
      */
      #randNonInputs(){ return _.randItem(this.#non_ins) }
      /**Create a new link with the probability of Params.chanceAddLink.
       * @param {number} mutationRate
       * @param {boolean} chanceOfLooped
       * @param {number} triesToFindLoop
       * @param {number} triesToAddLink
       */
      addLink(mutationRate, chanceOfLooped, triesToFindLoop, triesToAddLink){
        _.assert(triesToFindLoop>=0,"bad param: triesToFindLoop");
        _.assert(triesToAddLink>=0, "bad param: triesToAddLink");
        if(_.rand() > mutationRate){ return }
        let n1, n2, n, recurrent = false;
        //create link that loops back into the same neuron?
        if(_.rand() < chanceOfLooped){
          while(triesToFindLoop--){
            n=this.#randNonInputs();
            if(!n.recur){
              n.setRecur(recurrent = true);
              n1 = n2 = n;
              break;
            }
          }
        }else{
          while(triesToAddLink--){
            n2 = this.#randNonInputs();
            n1 = this.#randAny();
            if(n1.id == n2.id ||
               this.#dupLink(n1.id, n2.id)){
              n1 = n2 = UNDEF; // bad
            }else{
              break;
            }
          }
        }
        if(n1 && n2){
          if(n1.posY > n2.posY){ recurrent = true }
          if(this.#db.check(n1.id, n2.id, InnovType.LINK) < 0){
            this.#db.create(n1.id, n2.id, InnovType.LINK)
          }
          this.#vecLinks.push(new LinkGene(n1.id, n2.id, true, _.randMinus1To1(), recurrent));
          this.#sortGenes();
          //_.log(`addLink: gid(${this.#genomeID}): ${this.dbgState()}`);
        }
      }
      /**Adds a neuron to the genotype by examining the network,
       * splitting one of the links and inserting the new neuron.
       * @param {number} mutationRate
       * @param {number} triesToFindOldLink
       */
      addNeuron(mutationRate, triesToFindOldLink){
        _.assert(triesToFindOldLink>=0,"bad param: triesToFindOldLink");
        if(_.rand() > mutationRate){ return }
        //If the genome is small the code makes sure one of the older links is
        //split to ensure a chaining effect does not occur.
        //Here, if the genome contains less than 5 hidden neurons it
        //is considered to be too small to select a link at random
        let toID, fromID=-1,
            fLink, newNeuronID,
            numGenes=this.numGenes(),
            //bias towards older links
            offset=numGenes-1-int(Math.sqrt(numGenes)),
            _findID= (fLink)=> (fLink.enabled && !fLink.recur &&
                                !_isBIAS(this.#findNeuron(fLink.fromNeuron))) ? fLink.fromNeuron : -1;
        if(numGenes < this.#inputs+this.#outputs+5){
          while(fromID<0 && triesToFindOldLink--){
            fLink = this.#vecLinks[_.randInt2(0, offset)];
            fromID= _findID(fLink);
          }
        }else{
          while(fromID<0){
            fLink = _.randItem(this.#vecLinks);
            fromID=_findID(fLink);
          }
        }

        if(fromID<0){
          return;
        }

        _.assert(fLink, "addNeuron: unexpected null link gene!");
        fLink.setEnabled(false);
        toID=fLink.toNeuron;

        _.assert(fromID>0 && toID>0, `addNeuron: bad neuron ids: fromID: ${fromID}, toID: ${toID}`);

        //keep original weight so that the split does not disturb
        //anything the NN may have already learned...
        let oldWeight = fLink.weight,
            toObj=this.#findNeuron(toID),
            fromObj=this.#findNeuron(fromID),
            newPOS=_splitBetween(fromObj,toObj),
            iid = this.#db.check(fromID, toID, InnovType.NEURON);
        if(iid>0 && this.#hasNeuron(this.#db.getNeuronID(iid))){ iid=-1 }
        if(iid<0){
          //_.log(`addNeuron: need to create 2 new innovs`);
          newNeuronID= this.#db.create(fromID, toID,
                                       InnovType.NEURON,
                                       NeuronType.HIDDEN, newPOS).neuronID;
          _.assert(newNeuronID>0,`addNeuron: (+) unexpected -ve neuron id ${newNeuronID}`);
          //new innovations
          this.#db.create(fromID, newNeuronID, InnovType.LINK);
          this.#db.create(newNeuronID, toID, InnovType.LINK);
        }else{
          //_.log(`addNeuron: innov already exist or neuron added already`);
          //this innovation exists, find the neuron
          newNeuronID = this.#db.getNeuronID(iid);
          _.assert(newNeuronID>0,`addNeuron: (x) unexpected -ve neuron id ${newNeuronID}`);
        }

        //double check...
        _.assert(this.#db.check(fromID, newNeuronID, InnovType.LINK) >0 &&
                 this.#db.check(newNeuronID, toID, InnovType.LINK) >0, "addNeuron: expected innovations");

        //now we need to create 2 new genes to represent the new links
        this.#vecNeurons.push(new NeuronGene(newNeuronID, NeuronType.HIDDEN, newPOS));
        this.#non_ins.push(this.#vecNeurons.at(-1));
        this.#vecLinks.push(new LinkGene(fromID, newNeuronID, true, 1),
                            new LinkGene(newNeuronID, toID, true, oldWeight));
        this.#sortGenes();
        //_.log(`addNeuron: gid(${this.#genomeID}): ${this.dbgState()}`);
      }
      /**Get neuron with this id.
       * @param {number} id
       * @return {number}
       */
      #findNeuron(id){
        let obj= this.#vecNeurons.find(n=> n.id==id);
        return obj  ? obj : _.assert(false, "Error in Genome::findNeuron");
      }
      /**
       * @param {number} neuronIn
       * @param {number} neuronOut
       * @return {boolean} true if the link is already part of the genome
       */
      #dupLink(neuronIn, neuronOut){
        return this.#vecLinks.some(k=> k.fromNeuron == neuronIn && k.toNeuron == neuronOut)
      }
      /**Tests to see if the parameter is equal to any existing neuron ID's.
       * @param {number} id
       * @return {boolean} true if this is the case.
       */
      #hasNeuron(id){
        //_.log(`hasNeuron: checking if genome has this neuron: ${id}`);
        return id > 0 ? this.#vecNeurons.some(n=> n.id == id) : false;
      }
      /**
       * @param {number} mutationRate
       * @param {number} probNewWeight the chance that a weight may get replaced by a completely new weight.
       * @param {number} maxPertubation the maximum perturbation to be applied
       */
      mutateWeights(mutationRate, probNewWeight, maxPertubation){
        this.#vecLinks.forEach(k=>{
          if(_.rand() < mutationRate)
            k.weight= _.rand()<probNewWeight ? _.randMinus1To1()
                                              : k.weight + _.randMinus1To1() * maxPertubation;
        })
      }
      /**Perturbs the activation responses of the neurons.
       * @param {number} mutationRate
       * @param {number} maxPertubation the maximum perturbation to be applied
       */
      mutateActivation(mutationRate, maxPertubation){
        this.#vecNeurons.forEach(n=>{
          if(_.rand() < mutationRate)
            n.activation += _.randMinus1To1() * maxPertubation;
        })
      }
      /**Find the compatibility of this genome with the passed genome.
       * @param {Genome} other
       * @return {number}
       */
      calcCompat(other){
        //travel down the length of each genome counting the number of
        //disjoint genes, the number of excess genes and the number of matched genes
        let g1=0,g2=0,
            id1,id2,k1,k2,
            numDisjoint= 0,
            numExcess = 0,
            numMatched = 0,
            sumWeightDiff = 0,
            curEnd=this.numGenes(),
            otherEnd=other.numGenes();

        while(g1<curEnd || g2<otherEnd){

          //genome2 longer so increment the excess score
          if(g1 >= curEnd){ ++g2; ++numExcess; continue; }
          //genome1 longer so increment the excess score
          if(g2 >= otherEnd){ ++g1; ++numExcess; continue; }

          k2=other.geneAt(g2);
          k1=this.geneAt(g1);
          id2 = this.#db.getIID(k2);
          id1 = this.#db.getIID(k1);

          if(id1 == id2){
            ++g1; ++g2; ++numMatched;
            sumWeightDiff += Math.abs(k1.weight - k2.weight);
          }else{
            ++numDisjoint;
            if(id1 < id2){ ++g1 }
            else if(id1 > id2){ ++g2 }
          }
        }

        let disjoint = 1,
            excess   = 1,
            matched  = 0.4,
            longest= Math.max(this.numGenes(),other.numGenes()),
            xxx= (excess * numExcess/longest) + (disjoint * numDisjoint/longest);

        return numMatched>0 ? xxx + (matched * sumWeightDiff/ numMatched) : xxx;
      }
      /**
      */
      #sortGenes(){
        if(1)
          this.#vecLinks.sort(_.comparator(_.SORT_ASC,a=>this.#db.getIID(a), b=>this.#db.getIID(b)));
        return this;
      }
      /**
      */
      _cpy(fit,id,adjScore,spawnCnt,species,nodes,links){
        this.#fitness=Params.fitFunc(fit.score());
        this.#vecNeurons=nodes.map(v=>v.clone());
        this.#vecLinks=links.map(v=>v.clone());
        this.#spawnCnt=spawnCnt;
        this.#adjScore=adjScore;
        this.#species=species;
        this.#genomeID=id;
        this.#segregate();
        return this;
      }
      /**
       * @return {Genome}
       */
      clone(){
        return new Genome(this.#db, -1)._cpy(
          this.#fitness,
          this.#genomeID,
          this.#adjScore,
          this.#spawnCnt,
          this.#species,
          this.#vecNeurons,
          this.#vecLinks
        )
      }
      /**
       * @param {number} gid
       */
      morph(gid){
        this.#genomeID=gid;
        if(this.numNeurons() < Params.maxNNetNeurons)
          this.addNeuron(Params.chanceAddNode, Params.findOldLink);
        //now there's the chance a link may be added
        this.addLink(Params.chanceAddLink,
                     Params.chanceRecurrent,
                     Params.findLoopedLink, Params.addLinkAttempts);
        //mutate the weights
        this.mutateWeights(Params.mutationRate,
                           Params.chanceSetWeight,
                           Params.maxWeightJiggle);
        this.mutateActivation(Params.activationMutation, Params.maxActivationJiggle);
        return this;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Species
     * @class
     */
    class Species{

      #speciesID;
      #stale;
      #age;
      #numSpawn;
      #vecMembers;
      #leader;
      #bestScore;

      /**
      */
      get bestScore() { return this.#bestScore }
      get id() { return this.#speciesID }
      get leader() { return this.#leader }
      get stale(){ return this.#stale }
      get age(){ return this.#age }

      /**
       * @param {number} sid
       * @param {Genome} org
       */
      constructor(sid, org){
        this.#bestScore= org.getScore();
        this.#leader= org.clone();
        this.#vecMembers= [org];
        this.#speciesID= sid;
        this.#numSpawn=0;
        this.#age=0;
        this.#stale=0;
      }
      /**Adjusts the fitness of each individual by first
       * examining the species age and penalising if old, boosting if young.
       * Then we perform fitness sharing by dividing the fitness
       * by the number of individuals in the species.
       * This ensures a species does not grow too large.
       */
      adjustScores(){
        this.#vecMembers.forEach((g,i,a)=>{
          i = g.getScore();
          if(this.#age < Params.youngBonusAge){
            //boost the fitness scores if the species is young
            i *= Params.youngFitnessBonus
          }
          if(this.#age > Params.oldAgeThreshold){
            //punish older species
            i *= Params.oldAgePenalty
          }
          //apply fitness sharing to adjusted fitnesses
          g.adjustScore( i/a.length);
        });
        return this;
      }
      /**Adds a new member to this species and updates the member variables accordingly
       * @param {Genome} g
       */
      addMember(g){
        if(g.getScore() > this.#bestScore){
          this.#bestScore = g.getScore();
          this.#leader = g.clone();
          this.#stale = 0;
        }
        g.species= this.#speciesID;
        this.#vecMembers.push(g);
        return this;
      }
      /**Clears out all the members from the last generation, updates the age and gens no improvement.
       */
      purge(){
        this.#vecMembers.length=0;
        this.#numSpawn = 0;
        ++this.#stale;
        ++this.#age;
        return this;
      }
      /**Simply adds up the expected spawn amount for each individual
       * in the species to calculate the amount of offspring
       * this species should spawn.
       */
      calcSpawnAmount(){
        return this.#numSpawn= this.#vecMembers.reduce((acc,g)=> acc + g.spawnCnt, 0)
      }
      /**Spawns an individual from the species selected at random
       * from the best Params::dSurvivalRate percent.
       * @return {Genome} a random genome selected from the best individuals
       */
      spawn(){
        let n,baby,z=this.#vecMembers.length;
        if(z == 1){
          baby = this.#vecMembers[0]
        }else{
          n = int(Params.survivalRate * z)-1;
          if(n<0)n=1;
          if(n>=z)n=z-1;
          baby = this.#vecMembers[ _.randInt2(0, n) ];
        }
        return baby.clone();
      }
      /**
       * @param {number} tries
       * @return {array} [a,b]
       */
      randPair(tries=5){
        _.assert(tries>=0, "bad param: tries must be positive");
        let rc,g1,g2,n,z=this.#vecMembers.length;
        if(z == 1){
          g1=this.#vecMembers[0];
        }else{
          n = int(Params.survivalRate * z)-1;
          if(n<0)n=1;
          if(n>=z)n=z-1;
          g1= this.#vecMembers[ _.randInt2(0, n) ];
          while(tries--){
            g2= this.#vecMembers[ _.randInt2(0, n) ];
            if(g1.id == g2.id){g2=UNDEF}else{ break }
          }
        }
        return g2 ? [g1, g2] : [g1, null];
      }
      /**
       * @return {number}
       */
      numToSpawn(){ return this.#numSpawn }
      /**
       * @return {number}
       */
      size(){ return this.#vecMembers.length }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**A recursive function used to calculate a lookup table of split depths.
     */
    function _splitDepths(low, high, depth, out){
      const span = high-low;
      out.push({val: low + span/2, depth: depth+1});
      if(depth > 6){
      }else{
        _splitDepths(low, low+span/2, depth+1, out);
        _splitDepths(low+span/2, high, depth+1, out);
      }
      return out;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**NeatGA
     * @class
     */
    class NeatGA{

      #SPECIES_COUNTER;
      #GENOME_COUNTER;

      #vecSpecies;
      #vecSplits;
      #vecBest;

      #generation;
      #vecGenomes;
      #totFitAdj;
      #avFitAdj;
      #bestFit;
      #popSize;
      #innovDB;

      /**Creates a base genome from supplied values and creates a population
       * of 'size' similar (same topology, varying weights) genomes.
       * @param {number} size
       * @param {number} inputs
       * @param {number} outputs
       */
      constructor(size, inputs, outputs){
        //this holds the precalculated split depths. They are used
        //to calculate a neurons x/y position for rendering and also
        //for calculating the flush depth of the network when a
        //phenotype is working in 'snapshot' mode.
        //create the network depth lookup table
        this.#vecSplits = _splitDepths(0, 1, 0, []);
        this.#vecBest=[];
        this.#vecSpecies=[];

        this.#SPECIES_COUNTER=0;
        this.#GENOME_COUNTER=0;

        this.#generation=0;
        this.#popSize=size;

        //adjusted fitness scores
        this.#totFitAdj=0;
        this.#avFitAdj=0;
        this.#bestFit=0;

        this.#innovDB= new InnovHistory(inputs, outputs);
        this.#vecGenomes= _.fill(size, ()=> new Genome(this.#innovDB, this.#genGID()));
        //_.log(`NeatGA: inited population with ${this.#vecGenomes.length} genomes`);
      }
      /**
       * @return {number} current generation
       */
      curGen(){
        return this.#generation
      }
      /**
       * @param {Genome} mum
       * @param {Genome} dad
       */
      #crossOver(mum, dad){
        let mumEnd=mum.numGenes(), dadEnd=dad.numGenes();
        let mumIsBest = mum.getScore()>=dad.getScore();
        //step through each parents genes until we reach the end of both
        let bbGenes=[], bbNodes=[],
            curMum=0,curDad=0,chosen, pm, pd, iid1, iid2;
        while(!(curMum == mumEnd && curDad == dadEnd)){
          if(curMum == mumEnd){
            if(!mumIsBest){chosen = dad.geneAt(curDad)}
            ++curDad;
          }else if(curDad == dadEnd){
            if(mumIsBest){chosen = mum.geneAt(curMum)}
            ++curMum;
          }else{
            pm=mum.geneAt(curMum);
            pd=dad.geneAt(curDad);
            iid1=this.#innovDB.getIID(pm);
            iid2=this.#innovDB.getIID(pd);
            if(iid1 < iid2){
              if(mumIsBest){chosen = pm}
              ++curMum;
            }else if(iid2 < iid1){
              if(!mumIsBest){chosen = pd}
              ++curDad;
            }else{
              chosen=_.randAorB(pm,pd);
              ++curMum;
              ++curDad;
            }
          }
          _.assert(chosen, "crossOver: unexpected null as chosen");
          //add the selected gene if not already added
          if(bbGenes.length == 0 ||
             this.#innovDB.getIID(bbGenes.at(-1)) != this.#innovDB.getIID(chosen)){
            bbGenes.push(chosen.clone())
          }
          //Check if we already have the nodes referred to in chosen,
          //if not, they need to be added.
          [chosen.fromNeuron, chosen.toNeuron].forEach(nid=> bbNodes.indexOf(nid)<0 ? bbNodes.push(nid) : 0);
        }

        chosen= new Genome(this.#innovDB,-1)._specialClone(
                  this.#genGID(), bbNodes.sort().map(n=> this.#innovDB.createNeuronFromID(n)), bbGenes);
        //_.log(`crossOver: mum(${mum.id})+dad(${dad.id}): new-child:(${chosen.id}) with ${chosen.dbgState()}`);
        return chosen;
      }
      #genSID(){ return ++this.#SPECIES_COUNTER}
      #genGID(){ return ++this.#GENOME_COUNTER }
      /**Select NumComparisons members from the population at random testing
       * against the best found so far.
       * @param {number} howMany
       * @return {Genome}
       */
      tournamentSelection(howMany){
        let chosen,
            g, bestSoFar = 0;
        _.assert(howMany>=0, `tournamentSelection: bad arg value: ${howMany}`);
        while(howMany--){
          g = _.randItem(this.#vecGenomes);
          if(g.getScore() > bestSoFar){
            chosen = g;
            bestSoFar = g.getScore();
          }
        }
        return chosen;
      }
      /**Searches the lookup table for the splitY value of each node
       * in the genome and returns the depth of the network based on this figure.
       * @param {Genome} gen
       * @return {number}
       */
      calcNetDepth(gen){
        let maxSoFar = 0;
        for(let nd=0; nd<gen.numNeurons(); ++nd){
          for(let i=0; i<this.#vecSplits.length; ++i)
            if(gen.neuronAt(nd).posY == this.#vecSplits[i].val &&
               this.#vecSplits[i].depth > maxSoFar){
              maxSoFar = this.#vecSplits[i].depth;
            }
        }
        return maxSoFar + 2;
      }
      /**Separates each individual into its respective species by calculating
       * a compatibility score with every other member of the population and
       * niching accordingly. The function then adjusts the fitness scores of
       * each individual by species age and by sharing and also determines
       * how many offspring each individual should spawn.
       */
      #speciate(){
        this.#vecGenomes.forEach((g,i)=>{
          i= this.#vecSpecies.find(s=> g.calcCompat(s.leader) <= Params.compatThreshold);
          if(i){
            i.addMember(g);
          }else{
            this.#vecSpecies.push(new Species(this.#genSID(), g));
          }
        });
        //now that all the genomes have been assigned a species the fitness scores
        //need to be adjusted to take into account sharing and species age.
        this.#vecSpecies.forEach(s=> s.adjustScores())
        //calculate new adjusted total & average fitness for the population
        this.#vecGenomes.forEach(g=> this.#totFitAdj += g.adjScore);
        //////
        this.#avFitAdj = this.#totFitAdj / this.#vecGenomes.length;
        //calculate how many offspring each member of the population should spawn
        this.#vecGenomes.forEach(g=> g.spawnCnt=g.adjScore / this.#avFitAdj);
        //calculate how many offspring each species should spawn
        this.#vecSpecies.forEach(s=> s.calcSpawnAmount());
        //so we can sort species by best fitness. Largest first
        this.#vecSpecies.sort(_.comparator(_.SORT_DESC, a=>a.bestScore, b=>b.bestScore));
      }
      /**Sorts the population into descending fitness, keeps a record of the best
       * n genomes and updates any fitness statistics accordingly.
       */
      #sortAndRecord(scores){
        this.#vecGenomes.forEach((g,i)=> g.setScore(scores[i]));
        this.#vecGenomes.sort(_.comparator(_.SORT_DESC, a=>a.getScore(), b=>b.getScore()));
        this.#bestFit = Math.max(this.#bestFit,this.#vecGenomes[0].getScore());
        //save the best
        this.#vecBest.length=0;
        for(let i=0; i<Params.numBestElites; ++i)
          this.#vecBest.push(this.#vecGenomes[i]);
        return this;
      }
      /**Resets some values ready for the next epoch,
       * kills off all the phenotypes and any poorly performing species.
      */
      #resetAndKill(){
        this.#totFitAdj = 0;
        this.#avFitAdj  = 0;
        let L,tmp=[];
        this.#vecSpecies.forEach(s=>{
          if(s.stale > Params.noImprovements && s.bestScore < this.#bestFit){}else{
            tmp.push(s.purge());
          }
        });
        _.append(this.#vecSpecies, tmp, true);
        return this;
      }
      /**Performs one epoch of the genetic algorithm and returns a vector of pointers to the new phenotypes.
       * @param {number[]} scores
       * @return {}
       */
      epoch(scores){
        _.assert(scores.length == this.#vecGenomes.length, "NeatGA::Epoch(scores/ genomes mismatch)!");
        /////begin cleanse ////////////////////////////////////////////////////////////////////////////////////
        //1. reset appropriate values and kill off the existing phenotypes and any poorly performing species
        //2. update and sort genomes and keep a record of the best performers
        //3. separate the population into species of similar topology,
        this.#resetAndKill() && this.#sortAndRecord(scores) && this.#speciate();
        /////end cleanse /////////////////////////////////////////////////////////////////////////////////////
        let baby2,baby,newPop=[],numSpawnedSoFar=0;
        this.#vecSpecies.forEach(spc=>{
          if(numSpawnedSoFar<this.#popSize){
            let chosenBest= false,
                rc, count=_.rounded(spc.numToSpawn());
            while(count--){
              if(!chosenBest){
                chosenBest=true; baby=spc.leader.clone();
              }else if(spc.size() == 1 || _.rand() > Params.crossOverRate){
                //no crossOver
                baby = spc.spawn()
              }else{
                let [g1,g2] = spc.randPair(5);
                baby=g2 ? this.#crossOver(g1,g2) : g1.clone();
              }
              newPop.push(baby.morph(this.#genGID()));
              if(++numSpawnedSoFar == this.#popSize){ break }
            }
          }
        });
        if(numSpawnedSoFar < this.#popSize){
          let diff=this.#popSize - numSpawnedSoFar;
          while(diff--)
            newPop.push(this.tournamentSelection(int(this.#popSize/5)).clone());
        }
        //replace the current population with the new one
        _.append(this.#vecGenomes,newPop,true);
        ++this.#generation;
        //_.log(`NeatGA: current bestFitness = ${this.#bestFit}`);
        return this.createPhenotypes();
      }
      /**Cycles through all the members of the population and creates their phenotypes.
       * @return {NeuralNet[]} the new phenotypes
       */
      createPhenotypes(){
        return this.#vecGenomes.map(g=> g.createPhenotype( this.calcNetDepth(g)))
      }
      /**
       * @return {number}
       */
      numSpecies(){ return this.#vecSpecies.length }
      /**
       * @return {NeuralNet[]} the n best phenotypes from the previous generation.
       */
      bestFromPrevGen(){
        return this.#vecBest.map(g=> g.createPhenotype(this.calcNetDepth(g)))
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      NeatGA, NeuralNet, Genome, NeuronGene, LinkGene, NLink, NNeuron, Species,
      FitFunc, InnovHistory, NeuronType, InnovType, RunType,
      configParams(options){
        return _.inject(Params,options)
      }
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"))
  }else{
    gscope["io/czlab/mcfud/algo/NEAT_Buckland"]=_module
  }

})(this)



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

  console.log(`@czlab/crafty version: 1.5.0`);

})(this);


