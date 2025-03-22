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


