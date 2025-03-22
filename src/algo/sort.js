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

    const Basic= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/algo/basic"]();
    const Core= Mcfud ? Mcfud["Basic"] : gscope["io/czlab/mcfud/core"]();

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


