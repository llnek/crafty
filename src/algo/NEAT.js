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

    const {NodeType,FuncType,FuncTypeDB}= gscope["io/czlab/mcfud/algo/NNet"]();
    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();
    const _M= Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();
    const int=Math.floor;
    const {u:_, is}= Core;

    /**
     * @module mcfud/algo/NEAT
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @typedef {object} InnovType
     * @property {number} NODE
     * @property {number} LINK
     */
    const InnovType={ NODE:2, LINK:1 }

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
      BIAS: 1,//-1,
      //number of times we try to find 2 unlinked nodes when adding a link.
      addLinkAttempts:5,
      //number of attempts made to choose a node that is not an input
      //node and that does not already have a recurrently looped connection to itself
      findLoopedLink: 5,
      //the number of attempts made to find an old link to prevent chaining in addNode
      findOldLink: 5,
      //the chance, each epoch, that a node or link will be added to the genome
      probAddLink:0.07,
      probAddNode:0.03,
      chanceRecurrent: -1,//0.05,
      probCancelLink: 0.75,
      //mutation probabilities for mutating the weights
      mutationRate:0.8,
      maxWeightJiggle:0.5,
      probSetWeight:0.1,
      //probabilities for mutating the activation response
      activationMutation:0.1,
      maxActivationJiggle:0.1,
      //the smaller the number the more species will be created
      compatThreshold:0.26,
      //during score adjustment this is how much the fitnesses of
      //young species are boosted (eg 1.2 is a 20% boost)
      youngFitnessBonus:1.3,
      //if the species are below this age their fitnesses are boosted
      youngBonusAge:10,
      //number of population to survive each epoch. (0.2 = 20%)
      survivalRate:0,
      //if the species is above this age their score gets penalized
      oldAgeThreshold:50,
      //by this much
      oldAgePenalty:0.7,
      crossOverRate:0.7,
      //how long we allow a species to exist without any improvement
      noImprovements:15,
      //maximum number of neurons permitted in the network
      maxMeshNodes:100,
      numBestElites:4,
      actFunc: "sigmoid",
      fitFunc: function(seed=0){ return new ScoreFunc(seed) },
    };

    ////////////////////////////////////////////////////////////////////////////
    function _isOUTPUT(n){ return n.nodeType == NodeType.OUTPUT }
    function _isBIAS(n){ return n.nodeType == NodeType.BIAS }
    function _isINPUT(n,bias=false){
      return n.nodeType == NodeType.INPUT || (bias && n.nodeType==NodeType.BIAS);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function _calcSplits(inputs,outputs){
      return [ 1/(inputs+2), 1/(outputs+1) ]
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @param {NodeGene} from
     * @param {NodeGene} to
     * @return {Coord}
     */
    function _splitBetween(from,to){
      _.assert(from && to, `splitBetween: unexpected null params: from: ${from}, to: ${to}`);
      return new Coord((from.posX + to.posX)/2, (from.posY + to.posY)/2)
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class ScoreFunc{
      #value;
      constructor(seed){ this.#value=seed; }
      update(v){ this.#value=v; return this; }
      score(){ return this.#value }
      clone(){ return new ScoreFunc(this.#value) }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Coord{
      #x;
      #y;
      get x(){ return this.#x }
      get y(){ return this.#y }
      constructor(x=0,y=0){ this.#x=x; this.#y=y; }
      toJSON(){ return {x: this.x, y: this.y}}
      clone(){ return new Coord(this.#x, this.#y) }
      static dft(){ return new Coord(0,0) }
      static fromJSON(j){ return new Coord(j.x, j.y) }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NodeGene{
      #activation;
      #recurrent;
      #actFunc;
      #type;
      #pos;
      #id;
      /**
      */
      get activation(){ return this.#activation }
      get nodeType(){ return this.#type }
      get recur(){ return this.#recurrent }
      get actFunc(){ return this.#actFunc }
      get id(){ return this.#id }
      get pos() { return this.#pos }
      get posY(){ return this.#pos.y }
      get posX(){ return this.#pos.x }
      set activation(a){ this.#activation=a }
      /**
       * @param {number} id
       * @param {NodeType} type
       * @param {Coord} pos
       * @param {boolean} recur
       */
      constructor(id, type, pos=null,recur=false){
        _.assert(id>0, `creating a node with a bad id ${id}`);
        this.#pos= pos ? pos.clone() : Coord.dft();
        this.#recurrent= (recur===true);
        this.#activation=1;
        this.#id=id;
        this.#type= type;
      }
      /**
      */
      setActivation(a){
        this.#activation=a; return this;}
      /**
      */
      setActFunc(a){
        this.#actFunc=a; return this;
      }
      /**
      */
      setRecur(r){
        this.#recurrent=r; return this; }
      /**
      */
      eq(other){ return this.id==other.id }
      /**
      */
      prn(){
        return `${NodeType.toStr(this.nodeType)}#[${this.id}]` }
      /**
      */
      toJSON(){
        return {
          id: this.id,
          nodeType: this.nodeType,
          pos: this.pos.toJSON(),
          recur: this.recur,
          activation: this.activation,
          actFunc: is.str(this.actFunc) ? this.actFunc : ""
        }
      }
      /**
       */
      clone(){
        return new NodeGene(this.id,this.nodeType, this.pos, this.recur).
          setActFunc(this.actFunc).
          setActivation(this.activation);
      }
      /**
      */
      static fromJSON(j){
        return new NodeGene(j.id,j.nodeType,Coord.fromJSON(j.pos), j.recur).
          setActFunc(j.actFunc).
          setActivation(j.activation);
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class LinkGene{
      #recurrent;
      #enabled;
      #weight;
      #fromID;
      #toID;
      /**
      */
      get fromID(){ return this.#fromID }
      get toID(){ return this.#toID }
      get enabled(){ return this.#enabled }
      get weight(){ return this.#weight }
      get recur(){ return this.#recurrent }
      set weight(w){ this.#weight=w }
      /**
       * @param {number} from
       * @param {number} to
       * @param {boolean} enable
       * @param {number} w
       * @param {boolean} rec
       */
      constructor(from, to, enable=true, w=null, recur=false){
        this.#fromID= from;
        this.#toID= to;
        this.#recurrent=(recur===true);
        this.#enabled=(enable !== false);
        this.#weight= w===null||isNaN(w) ? _.randMinus1To1() : w;
      }
      /**
      */
      eq(other){
        return other.fromID == this.fromID && other.toID == this.toID
      }
      /**
      */
      clone(){
        return new LinkGene(this.#fromID,this.#toID,
                            this.#enabled, this.#weight, this.#recurrent) }
      /**
       */
      setRecur(r){
        this.#recurrent=r; return this; }
      /**
       */
      setEnabled(e){
        this.#enabled=e; return this; }
      /**
      */
      toJSON(){
        return {
          fromID: this.fromID,
          toID: this.toID,
          recur: this.recur,
          weight: this.weight,
          enabled: this.enabled
        }
      }
      /**
      */
      static fromJSON(j){
        return new LinkGene(j.fromID, j.toID, j.enabled, j.weight, j.recur) }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Innovation is a particular change to a Genome's structure. Each time a
     * genome undergoes a change, that change is recorded as an innovation and
     * is stored in a global historical database.
     * @class
     */
    class Innov{
      #innovType;
      #nodeType;
      #innovID;
      #nodeID;
      #fromID;
      #toID;
      #pos;
      /**
      */
      get innovType(){ return this.#innovType }
      get nodeID(){ return this.#nodeID }
      get IID(){ return this.#innovID }
      get pos() { return this.#pos }
      get fromID(){ return this.#fromID }
      get toID(){ return this.#toID }
      get nodeType(){ return this.#nodeType }
      /**
       * @param {InnovDB} db
       * @param {number} from
       * @param {number} to
       * @param {InnovType} type
       * @param {array} extra [id,NodeType]
       * @param {Coord} pos
       */
      constructor(db, from, to, type, extra=null, pos=null){
        this.#pos= pos ? pos.clone() : Coord.dft();
        this.#innovID= db.genIID();
        this.#innovType=type;
        this.#fromID= from;
        this.#toID= to;
        if(is.vecN(extra,2,true)){
          this.#nodeType= extra[1];
          this.#nodeID= extra[0];
        }else{
          this.#nodeID= -31;
          this.#nodeType= NodeType.NONE;
        }
        db.add(this);
      }
      /**
      */
      toJSON(){
        return {
          pos: this.pos.toJSON(),
          type: this.innovType,
          id: this.IID,
          fromID: this.fromID,
          toID: this.toID,
          nodeID: this.nodeID,
          nodeType: this.nodeType
        }
      }
      /**
      */
      static fromJSON(j,db){
        return new Innov(db, j.fromID, j.toID, j.type, [j.nodeID, j.nodeType], j.pos)
      }
      /**
       * @param {InnovDB} db
       * @param {number} nid node id
       * @param {NodeType} type
       * @param {Coord} pos
       * @return {Innov}
      */
      static from(db, nid, type, pos){
        return new Innov(db, -71,-99, InnovType.NODE, [nid, type], pos) }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Used to keep track of all innovations created during the populations
     * evolution, adds all the appropriate innovations.
     * @class
     */
    class InnovDB{
      #innovCounter;
      #vecInnovs;
      #topology;
      /**
      */
      get parent(){ return this.#topology }
      /**Initialize the history database.
       * @param {Topology} t
       */
      constructor(t){
        this.#innovCounter=0;
        this.#vecInnovs=[];
        this.#topology=t;
      }
      /**
       * @return {number} next innovation number
       */
      genIID(){ return ++this.#innovCounter }
      /**Checks to see if this innovation has already occurred. If it has it
       * returns the innovation ID. If not it returns a negative value.
       * @param {number} from
       * @param {number} out
       * @param {InnovType} type
       * @return {number}
       */
      check(from, out, type){
        _.assert(from>0 && out>0, `checking innov with bad node ids: from: ${from}, to: ${out}`);
        const rc= this.#vecInnovs.find(cur=> cur.innovType == type &&
                                             cur.fromID == from && cur.toID == out);
        return rc ? rc.IID : -51;
      }
      /**
       * @param {Innov} n
       */
      add(n){
        this.#vecInnovs.push(n); return this; }
      /**Creates a new innovation.
       * @param {number} from
       * @param {number} to
       * @param {InnovType} innovType
       * @param {NodeType} nodeType
       * @param {Coord} pos
       * @return {Innov}
       */
      create(from, to, innovType, nodeType=NodeType.NONE, pos=null){
        let i;
        if(innovType==InnovType.NODE){
          _.assert(nodeType != NodeType.NONE, "create-innov: unexpected bad neuron type");
          _.assert(from>0&&to>0, `create-innov: bad neuron ids: from: ${from} to: ${to}`);
          i= new Innov(this, from, to, innovType, [this.parent.genNID(),nodeType], pos)
        }else{
          i= new Innov(this, from, to, innovType, null, pos);
        }
        return i;
      }
      /**
       * @param {number} iid innov id
       * @return {number} Node ID or -1
      */
      getNodeID(iid){
        const rc= this.#vecInnovs.find(n=> n.IID == iid);
        return rc ? rc.nodeID : -41;
      }
      /**
       * @param {LinkGene} gene
       * @param {InnovType} type
       * @return {number} innov id.
       */
      getIID(gene, type=InnovType.LINK){
        return this.check(gene.fromID, gene.toID, type)
      }
      /**
       * @param {LinkGene} gene
       * @param {InnovType} type
       * @return {Innov} innov
       */
      getInnov(gene, type=InnovType.LINK){
        return this.#vecInnovs.find(i=> i.innovType == type &&
                                        i.fromID == gene.fromID && i.toID == gene.toID)
      }
      /**
      */
      findInnovWithIID(iid){
        return this.#vecInnovs.find(i=> i.IID==iid);
      }
      /**
      */
      getInnovWithNodeID(nid){
        return this.#vecInnovs.find(n=> n.nodeID==nid);
      }
      /**
      */
      toJSON(){
        return {
          counter: this.#innovCounter,
          innovs: this.#vecInnovs.map(i=> i.toJSON())
        }
      }
      /**
      */
      _fromJSON(j){
        this.#innovCounter= j.counter;
        this.#vecInnovs= j.innovs.map(o=> Innov.fromJSON(o));
      }
      /**
      */
      static fromJSON(j, t){
        return new InnovDB(t)._fromJSON(j)
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Link{
      #weight;
      #from;
      #out;
      #recur;
      /**
      */
      get weight(){ return this.#weight }
      get from(){ return this.#from }
      get recur(){ return this.#recur }
      /**
       * @param {number} w
       * @param {Node} from
       * @param {Node} out
       * @param {boolean} recur
       */
      constructor(w, from, out, recur=false){
        this.#weight=w;
        this.#from=from;
        this.#out=out;
        this.#recur= (recur===true);
      }
      /**
       */
      clone(){
        return new Link(this.#weight, this.#from, this.#out, this.#recur)
      }
      /**
       * Create a link between these two neurons and
       * assign the weight stored in the gene.
       * @param {LinkGene} gene
       * @param {Node} from
       * @param {Node} to
       * @return {Link}
       */
      static from(gene,from,to){
        const rc= new Link(gene.weight, from, to, gene.recur);
        //add new links to nodes
        from.addLinkOut(rc);
        to.addLinkIn(rc);
        return rc;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Node{
      #vecLinksOut;
      #activation;
      #vecLinksIn;
      #nodeType;
      #nodeID;
      #output;
      #pos;
      #actFunc;
      /**
      */
      get activation() { return this.#activation }
      get nodeType() { return this.#nodeType }
      get id() { return this.#nodeID }
      get pos(){ return this.#pos }
      get posY(){ return this.#pos.y }
      get actFunc(){ return this.#actFunc }
      get outputValue() { return this.#output }
      set outputValue(o) { this.#output=o }
      /**
       * @param {number} id
       * @param {NodeType} type
       * @param {Coord} pos
       * @param {number} act_response
       */
      constructor(id,type, pos=null, act_response=1){
        this.#pos= pos ? pos.clone() : Coord.dft();
        this.#activation=act_response;
        this.#nodeType=type;
        this.#nodeID=id;
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
       */
      prn(){ return `node(${NodeType.toStr(this.nodeType)})#[${this.id}]`; }
      /**
      */
      flush(){
        this.outputValue=0; return this; }
      /**
      */
      clone(){
        return new Node(this.id,this.nodeType,this.pos,this.activation).
               _cpy(this.outputValue,this.#vecLinksIn, this.#vecLinksOut)
      }
      /**
      */
      setActFunc(a){
        this.#actFunc=a; return this; }
      /**
       * @param {Function} func
       * @return {any}
       */
      funcOverInLinks(func){ return func(this.#vecLinksIn) }
      /**
       * @param {Link} n
       */
      addLinkIn(n){
        this.#vecLinksIn.push(n); return this; }
      /**
       * @param {Link} o
       * @return {Node} this
       */
      addLinkOut(o){
        this.#vecLinksOut.push(o); return this; }
      /**
       * @param {NodeGene} n
       * @return {Node}
       */
      static from(n){
        return new Node(n.id, n.nodeType, n.pos, n.activation)
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NodeMesh{
      #vecNodes;
      #depth;
      /**
      */
      get depth(){return this.#depth}
      /**
       * @param {Node[]} nodes
       */
      constructor(nodes){
        _.append(this.#vecNodes=[], nodes, true);
        this.#calcDepth();
      }
      #calcDepth(){
        this.#depth= _.groupSimilar(this.#vecNodes.map(n=> n.posY), _.feq).length;
        //_.log(`depth==== ${this.#depth}`);
      }
      /**
       */
      clone(){
        return new NodeMesh(this.#vecNodes.map(n=>n.clone())) }
      /**Update mesh for this clock cycle.
       * @param {number[]} inputs
       * @param {RunType} type
       * @return {number[]} outputs
       */
      compute(inputs,type=RunType.ACTIVE){
        return this.update(inputs, type)
      }
      /**Update mesh for this clock cycle.
       * @param {number[]} inputs
       * @param {RunType} type
       * @return {number[]} outputs
       */
      update(inputs, type=RunType.ACTIVE){
        //if the mode is snapshot then we require all the nodes to be
        //iterated through as many times as the network is deep. If the
        //mode is set to active the method can return an output after just one iteration
        let arr,outputs=[],
            loopCnt= type==RunType.SNAPSHOT ? this.depth : 1;
        function _sum(a){
          return a.reduce((acc,k)=> acc + k.weight * k.from.outputValue,0);
        }
        while(loopCnt--){
          outputs.length=0;
          arr=this.#vecNodes.filter(n=> _isINPUT(n));
          _.assert(arr.length<=inputs.length, `NodeMesh: update with mismatched input size ${inputs.length}`);
          arr.forEach((n,i)=>{ n.outputValue=inputs[i] });
          this.#vecNodes.find(n=> _isBIAS(n)).outputValue= Params.BIAS;
          //now deal with the other nodes...
          this.#vecNodes.forEach(obj=>{
            if(!_isINPUT(obj,true)){
              let fn=obj.actFunc || Params.actFunc;
              if(!is.fun(fn)) fn=FuncTypeDB[fn || ""];
              if(!fn) fn= FuncTypeDB["sigmoid"];
              obj.outputValue = fn(obj.funcOverInLinks(_sum)/obj.activation);
              if(_isOUTPUT(obj)) outputs.push(obj.outputValue);
            }
          });
        }

        if(type == RunType.SNAPSHOT){
          this.#vecNodes.forEach(n=> n.flush());
        }

        /////
        return outputs;
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
      #vecNodes;
      #vecLinks;
      #non_ins;
      #genomeID;
      #score;
      //its score score after it has been placed into a species and adjusted accordingly
      #adjScore;
      //the number of offspring is required to spawn for the next generation
      #spawnCnt;
      #inputs;
      #outputs;
      //keeps a track of which species this genome is in
      #species;
      #topology;
      /**
      */
      get spawnCnt() { return this.#spawnCnt }
      get adjScore(){ return this.#adjScore }
      get id(){ return this.#genomeID }
      get parent(){ return this.#topology }
      set spawnCnt(n) { this.#spawnCnt = n }
      /**A genome basically consists of a vector of link genes,
       * a vector of node genes and a score score.
       * @param {Topology} t
       * @param {boolean} huskOnly
       */
      constructor(t,huskOnly=false){
        this.#score=Params.fitFunc(0);
        this.#vecNodes=[];
        this.#vecLinks=[];
        this.#genomeID= -1;
        this.#topology=t;
        this.#species=0;
        this.#adjScore=0;
        this.#spawnCnt=0;
        this.#non_ins=[];
        if(!huskOnly){
          this.#genomeID= t.genGID();
          t.naissance(this).#segregate();
        }
      }
      /**
      */
      #segregate(){
        this.#non_ins= this.#vecNodes.reduce((acc,n)=>{ if(!_isINPUT(n,true)){acc.push(n)} return acc; }, []);
        this.#vecNodes.sort(_.comparator(_.SORT_ASC, a=> a.posY, b=> b.posY));
        return this.#sortGenes();
      }
      /**
       */
      #prnNodes(){
        return this.#vecNodes.reduce((acc,n,i)=> acc += ((i==0)?"":", ") + n.prn(), "")
      }
      /**
      */
      #dbgCtor(){
        if(0 && this.id>0)
          _.log(`genome(${this.id}):${this.#prnNodes()}`);
      }
      /**
      */
      crossOverWith(other){
        let isEnabled = [], newGenes = [];
        this.#vecLinks.forEach((k,i)=>{
          let en = true;
          let p2= other.findInnov(this.parent.db.getIID(k));
          if(p2){
            _.assert(k.eq(p2), `expected links to be same, but not!, ${k.fromID} to ${k.toID}, and ${p2.fromID} to ${p2.toID}`);
            if(!k.enabled || !p2.enabled){
              if(_.rand() < Params.probCancelLink){
                en = false;
              }
            }
            newGenes.push(_.randAorB(k,p2));
          }else{
            //disjoint or excess gene
            en = k.enabled;
            newGenes.push(k);
          }
          isEnabled.push(en);
        });
        let gs,vs=[];
        newGenes.forEach((k,i)=>{
          i= this.#vecNodes.find(n=> k.fromID==n.id);
          _.assert(i, `unexpectedly missing node ${k.fromID} in crossOver`);
          if(!vs.find(n=> n.id == i.id)) vs.push(i);
          i= this.#vecNodes.find(n=> k.toID==n.id);
          _.assert(i, `unexpectedly missing node ${k.toID} in crossOver`);
          if(!vs.find(n=> n.id == i.id)) vs.push(i);
        });
        vs= vs.map(n=> n.clone());
        gs= newGenes.map((k,i)=> k.clone().setEnabled(isEnabled[i]));
        return new Genome(this.parent,true)._inflate(vs, gs);
      }
      /**
       */
      dbgState(){
        return `{nodes=${this.#prnNodes()},links=${this.#vecLinks.length}}`
      }
      /**
       * @param {number} n
       */
      adjustScore(n){
        this.#adjScore=n; return this; }
      /**
       * @return {number} number of neurons
       */
      size() { return this.#vecNodes.length }
      /**
       * @return {number} number of links
      */
      scale() { return this.#vecLinks.length }
      /**
       * @param {any} num score
       */
      setScore(num){
        this.#score.update(num); return this; }
      /**
       * @return {number}
       */
      getScore(){ return this.#score.score() }
      /**
       * @return {LinkGene}
       */
      geneAt(i) { return this.#vecLinks[i]  }
      /**
       * @return {NodeGene}
       */
      nodeAt(i) { return this.#vecNodes[i] }
      /**
       * @param {number} newid
       */
      mutateGID(newid){
        _.assert(newid>0, "bad genome id, must be positive"); this.#genomeID=newid; return this; }
      /**
      */
      _inflate(nodes,links){
        _.append(this.#vecNodes, nodes, true);
        _.append(this.#vecLinks, links, true);
        if(this.id<0)
          this.#genomeID= this.parent.genGID();
        return this.#segregate();
      }
      /**
      */
      findInnov(iid){
        let v= this.parent.db.findInnovWithIID(iid);
        return this.#vecLinks.find(k=> k.fromID == v.fromID && k.toID == v.toID);
      }
      /**Create a mesh from the genome.
       * @return {NodeMesh} newly created mesh
       */
      phenotype(){
        const vs= this.#vecNodes.map(g=> Node.from(g));
        this.#vecLinks.forEach(k=>
          k.enabled? Link.from(k, vs.find(n=> n.id== k.fromID),
                                   vs.find(n=> n.id== k.toID)) :0);
        return new NodeMesh(vs);
      }
      #randAny(){ return _.randItem(this.#vecNodes) }
      #randNonInputs(){ return _.randItem(this.#non_ins) }
      /**Create a new link with the probability of Params.probAddLink.
       * @param {number} mutationRate
       * @param {boolean} chanceOfLooped
       * @param {number} triesToFindLoop
       * @param {number} triesToAddLink
       */
      addLink(mutationRate, chanceOfLooped, triesToFindLoop, triesToAddLink){
        if(_.rand() < mutationRate){}else{ return }
        let n1, n2, n, recur= false;
        //create link that loops back?
        if(_.rand() < chanceOfLooped){
          triesToFindLoop=Math.min(1,triesToFindLoop);
          while(triesToFindLoop--){
            n=this.#randNonInputs();
            if(!n.recur){
              n.setRecur(recur=true);
              n1 = n2 = n;
              break;
            }
          }
        }else{
          triesToAddLink=Math.min(1,triesToAddLink);
          while(triesToAddLink--){
            n2 = this.#randNonInputs();
            n1 = this.#randAny();
            if(!n2 || !n1){
              throw "poo";
            }
            if(n1.id == n2.id ||
               this.#dupLink(n1.id, n2.id)){
              n1 = n2 = UNDEF; // bad
            }else{
              break;
            }
          }
        }
        if(n1 && n2){
          if(n1.posY > n2.posY){ recur=true }
          if(this.parent.db.check(n1.id, n2.id, InnovType.LINK) < 0){
            this.parent.db.create(n1.id, n2.id, InnovType.LINK)
          }
          this.#vecLinks.push(new LinkGene(n1.id, n2.id, true, _.randMinus1To1(), recur));
          //_.log(`addLink: gid(${this.#genomeID}): ${this.dbgState()}`);
        }
      }
      /**Adds a node to the genotype by examining the mesh,
       * splitting one of the links and inserting the new node.
       * @param {number} mutationRate
       * @param {number} triesToFindOldLink
       */
      addNode(mutationRate, triesToFindOldLink){
        if(_.rand() < mutationRate){}else{ return }
        //If the genome is small the code makes sure one of the older links is
        //split to ensure a chaining effect does not occur.
        //Here, if the genome contains less than 5 hidden nodes it
        //is considered to be too small to select a link at random
        let newNID, toID, fromID=-1,
            fLink, numGenes=this.scale(),
            //bias towards older links
            offset=numGenes-1-int(Math.sqrt(numGenes)),
            _findID= (k)=> (k.enabled && !k.recur && !_isBIAS(this.#findNode(k.fromID))) ? k.fromID : -1;
        triesToFindOldLink=Math.min(1,triesToFindOldLink);
        if(numGenes < this.parent.inSlots+this.parent.outSlots+5){
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

        _.assert(fLink, "addNode: unexpected null link gene!");
        fLink.setEnabled(false);
        toID=fLink.toID;

        _.assert(fromID>0 && toID>0, `addNode: bad node ids: fromID: ${fromID}, toID: ${toID}`);

        //keep original weight so that the split does not disturb
        //anything the genome may have already learned...
        let oldWeight = fLink.weight,
            toObj=this.#findNode(toID),
            fromObj=this.#findNode(fromID),
            newPOS=_splitBetween(fromObj,toObj),
            iid = this.parent.db.check(fromID, toID, InnovType.NODE);
        if(iid>0 && this.#hasNode(this.parent.db.getNodeID(iid))){ iid=-1 }
        if(iid<0){
          //_.log(`addNode: need to create 2 new innovs`);
          newNID= this.parent.db.create(fromID, toID,
                                        InnovType.NODE,
                                        NodeType.HIDDEN, newPOS).nodeID;
          _.assert(newNID>0,`addNode: (+) unexpected -ve neuron id ${newNID}`);
          //new innovations
          this.parent.db.create(fromID, newNID, InnovType.LINK);
          this.parent.db.create(newNID, toID, InnovType.LINK);
        }else{
          //_.log(`addNode: innov already exist or node added already`);
          //this innovation exists, find the neuron
          newNID = this.parent.db.getNodeID(iid);
          _.assert(newNID>0,`addNode: (x) unexpected -ve neuron id ${newNID}`);
        }

        //double check...
        _.assert(this.parent.db.check(fromID, newNID, InnovType.LINK) >0 &&
                 this.parent.db.check(newNID, toID, InnovType.LINK) >0, "addNode: expected innovations");

        //now we need to create 2 new genes to represent the new links
        this.#vecNodes.push(new NodeGene(newNID, NodeType.HIDDEN, newPOS));
        this.#vecLinks.push(new LinkGene(fromID, newNID, true, 1),
                            new LinkGene(newNID, toID, true, oldWeight));
        //_.log(`addNode: gid(${this.#genomeID}): ${this.dbgState()}`);
      }
      /**Get node with this id.
       * @param {number} id
       * @return {number}
       */
      #findNode(id){
        let obj= this.#vecNodes.find(n=> n.id==id);
        return obj  ? obj : _.assert(false, "Error in Genome::findNode");
      }
      /**
       * @param {number} fromID
       * @param {number} toID
       * @return {boolean} true if the link is already part of the genome
       */
      #dupLink(fromID, toID){
        return this.#vecLinks.some(k=> k.fromID == fromID && k.toID == toID)
      }
      /**Tests to see if the parameter is equal to any existing node ID's.
       * @param {number} id
       * @return {boolean} true if this is the case.
       */
      #hasNode(id){
        //_.log(`hasNode: checking if genome has this node: ${id}`);
        return id > 0 ? this.#vecNodes.some(n=> n.id == id) : false;
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
      /**Perturbs the activation responses of the nodes..
       * @param {number} mutationRate
       * @param {number} maxPertubation the maximum perturbation to be applied
       */
      mutateActivation(mutationRate, maxPertubation){
        this.#vecNodes.forEach(n=>{
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
            curEnd=this.scale(),
            otherEnd=other.scale();

        while(g1<curEnd || g2<otherEnd){
          //genome2 longer so increment the excess score
          if(g1 >= curEnd){ ++g2; ++numExcess; continue; }
          //genome1 longer so increment the excess score
          if(g2 >= otherEnd){ ++g1; ++numExcess; continue; }

          k2=other.geneAt(g2);
          k1=this.geneAt(g1);
          id2 = this.parent.db.getIID(k2);
          id1 = this.parent.db.getIID(k1);

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
            longest= Math.max(this.scale(),other.scale()),
            xxx= (excess * numExcess/longest) + (disjoint * numDisjoint/longest);

        return numMatched>0 ? xxx + (matched * sumWeightDiff/ numMatched) : xxx;
      }
      /**
      */
      #sortGenes(){
        this.#vecLinks.sort( _.comparator( _.SORT_ASC,
                                           a=>this.parent.db.getIID(a),
                                           b=>this.parent.db.getIID(b)));
        return this;
      }
      /**
      */
      _cpy(id,fit,adjScore,spawnCnt,species,nodes,links){
        this.#score=Params.fitFunc(fit.score());
        this.#vecNodes=nodes.map(v=>v.clone());
        this.#vecLinks=links.map(v=>v.clone());
        this.#spawnCnt=spawnCnt;
        this.#adjScore=adjScore;
        this.#species=species;
        this.#genomeID=id;
        return this.#segregate();
      }
      /**
       */
      clone(gid){
        return new Genome(this.parent, true)._cpy(
          gid || this.#genomeID,
          this.#score,
          this.#adjScore,
          this.#spawnCnt,
          this.#species,
          this.#vecNodes,
          this.#vecLinks
        )
      }
      /**
       */
      morph(){
        if(this.size() < Params.maxMeshNodes)
          this.addNode(Params.probAddNode, Params.findOldLink);
        //now there's the chance a link may be added
        this.addLink(Params.probAddLink,
                     Params.chanceRecurrent,
                     Params.findLoopedLink, Params.addLinkAttempts);
        //mutate the weights
        this.mutateWeights(Params.mutationRate,
                           Params.probSetWeight,
                           Params.maxWeightJiggle);
        this.mutateActivation(Params.activationMutation, Params.maxActivationJiggle);
        return this.#segregate();
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Topology{
      #genomeCounter;
      #speciesCounter;
      #nodeCounter;
      #vecNodes;
      #options;
      #inputs;
      #outputs;
      #db;
      get outSlots(){return this.#outputs}
      get inSlots(){return this.#inputs}
      get db(){ return this.#db }
      /**
      */
      constructor(inputs,outputs,options){
        this.#speciesCounter=0;
        this.#genomeCounter=0;
        this.#nodeCounter=0;
        this.#outputs=outputs;
        this.#inputs=inputs;
        this.#vecNodes=[];
        this.#db= new InnovDB(this);
        this.#doLayout(inputs, outputs, options || {});
      }
      #doLayout(inputs, outputs, options){
        let [iXGap, oXGap] = _calcSplits(inputs,outputs);
        let nObj, nid=0;
        for(let i=0; i<inputs; ++i){
          nObj={t: NodeType.INPUT, id: ++nid,  co: new Coord((i+2)*iXGap,0)};
          this.#vecNodes.push(nObj);
          Innov.from(this.db, nObj.id, nObj.t,  nObj.co);
        }

        nObj= {t:NodeType.BIAS, id: ++nid, co: new Coord(iXGap,0)};
        this.#vecNodes.push(nObj);
        Innov.from(this.db, nObj.id, nObj.t, nObj.co);

        for(let i=0; i<outputs; ++i){
          nObj={act: options.actOutFunc, t:NodeType.OUTPUT, id: ++nid, co: new Coord((i+1)*oXGap,1) };
          this.#vecNodes.push(nObj);
          Innov.from(this.db, nObj.id, nObj.t, nObj.co);
        }

        _.assert(nid==inputs+outputs+1,"bad layout - mismatched node ids");
        _.assert(nid==this.#vecNodes.at(-1).id, "bad layout - erroneous last node id");

        this.#nodeCounter= nid;
        this.#options=options;

        //connect each input & bias node to each output node
        if(1){
          let a= this.#vecNodes.filter(n=>n.t != NodeType.OUTPUT);
          let b= this.#vecNodes.filter(n=>n.t == NodeType.OUTPUT);
          a.forEach(i=> b.forEach(o => new Innov(this.db, i.id, o.id, InnovType.LINK)));
        }
      }
      /**
      */
      naissance(g){
        //make genes then connect each input & bias node to each output node
        let ins= this.#vecNodes.filter(nObj=>nObj.t != NodeType.OUTPUT);
        let os= this.#vecNodes.filter(nObj=>nObj.t == NodeType.OUTPUT);
        let nodes=[], links=[];
        this.#vecNodes.forEach((nObj,i)=>{
          i=new NodeGene(nObj.id, nObj.t, nObj.co);
          if(_isOUTPUT(i)) i.setActFunc(this.#options.actFuncOut);
          nodes.push(i);
        });
        ins.forEach(i=> os.forEach(o=> links.push(new LinkGene(i.id, o.id))));
        return g._inflate(nodes,links);
      }
      genSID(){ return ++this.#speciesCounter }
      /**
      */
      genGID(){ return ++this.#genomeCounter }
      /**
      */
      genNID(){ return ++this.#nodeCounter }
      /**
       * @param {number} nid Node Id.
       * @return {NodeGene}
       */
      createNodeFromID(nid){
        const i= this.db.getInnovWithNodeID(nid);
        _.assert(i, "unknown node id not found in innov history.");
        return new NodeGene(nid, i.nodeType, i.pos).setActFunc(this.#options.actFunc);
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Species
     * @class
     */
    class Species{
      #speciesID;
      #topology;
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
      get parent(){ return this.#topology }
      /**
       * @param {Topology} t
       * @param {Genome} org
       */
      constructor(t, org){
        this.#bestScore= org.getScore();
        this.#leader= org.clone();
        this.#vecMembers= [org];
        this.#speciesID= t.genSID();
        this.#numSpawn=0;
        this.#age=0;
        this.#stale=0;
        this.#topology=t;
      }
      /**Adjusts the score of each individual by first
       * examining the species age and penalising if old, boosting if young.
       * Then we perform score sharing by dividing the score
       * by the number of individuals in the species.
       * This ensures a species does not grow too large.
       */
      adjustScores(){
        this.#vecMembers.forEach((g,i,a)=>{
          i = g.getScore();
          if(this.#age < Params.youngBonusAge){
            //boost the score scores if the species is young
            i *= Params.youngFitnessBonus
          }
          if(this.#age > Params.oldAgeThreshold){
            //punish older species
            i *= Params.oldAgePenalty
          }
          //apply score sharing to adjusted fitnesses
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
        _.trunc(this.#vecMembers);
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
        return baby.clone(this.parent.genGID());
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
    /**NeatGA
     * @class
     */
    class NeatGA{
      #vecSpecies;
      #vecBest;
      #cur;
      #cycles;
      #totalScoreAdj;
      #avgScoreAdj;
      #bestScore;
      #popSize;
      #topology;
      /**Creates a base genome from supplied values and creates a population
       * of 'size' similar (same topology, varying weights) genomes.
       * @param {number} size
       * @param {number} inputs
       * @param {number} outputs
       * @param {object} options
       */
      constructor(size, inputs, outputs, options){
        this.#cycles=0;
        this.#popSize=size;
        this.#vecSpecies=[];
        this.#vecBest=[];
        this.#cur=[];
        //adjusted score scores
        this.#totalScoreAdj=0;
        this.#avgScoreAdj=0;
        this.#bestScore=0;
        this.#topology= new Topology(inputs, outputs, options);
      }
      #hatch(){
        return this.#cur= _.fill(this.#popSize, ()=> new Genome(this.#topology));
      }
      /**
       * @return {number} current generation
       */
      curGen(){
        return this.#cycles; }
      /**Performs one epoch of the genetic algorithm and
       * returns a vector of pointers to the new phenotypes.
       * @param {number[]} scores
       * @return {}
       */
      epoch(scores){
        _.assert(scores.length == this.#cur.length, "NeatGA::Epoch(scores/ genomes mismatch)!");
        let newPop=this.#cleanse(scores).#rejuvenate();
        let diff= this.#popSize- newPop.length;
        while(diff--)
          newPop.push(this.tournamentSelection(int(this.#popSize/5)).clone(this.#topology.genGID()));
        _.assert(newPop.length == this.#popSize, "NeatGA::Epoch(new genomes count mismatch)!");
        _.append(this.#cur,newPop,true);
        this.#cycles += 1;
        //_.log(`NeatGA: current bestFitness = ${this.#bestScore}`);
      }
      /**Cycles through all the members of the population and creates their phenotypes.
       * @return {NodeMesh[]} the new phenotypes
       */
      createPhenotypes(){
        return (this.#cur.length==0 ? this.#hatch() : this.#cur).map(g=> g.phenotype());
      }
      /**
       * @return {number}
       */
      numSpecies(){ return this.#vecSpecies.length }
      /**
       * @return {NodeMesh[]} the n best phenotypes from the previous generation.
       */
      bestFromPrevGen(){
        return this.#vecBest.map(g=> g.phenotype());
      }
      #crossOver(mum,dad){
        let p1,p2;
        if(mum.getScore()>dad.getScore()){
          p1=mum; p2=dad;
        }else if(mum.getScore()<dad.getScore()){
          p1=dad;p2=mum;
        }else if(_.randSign()>0){
           p1=mum; p2=dad;
        }else{
          p1=dad;p2=mum;
        }
        return p1.crossOverWith(p2);
      }
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
          g = _.randItem(this.#cur);
          if(g.getScore() > bestSoFar){
            chosen = g;
            bestSoFar = g.getScore();
          }
        }
        return chosen || this.#cur[0];
      }
      #rejuvenate(){
        let baby2,baby,newPop=[];
        this.#vecSpecies.forEach(spc=>{
          if(newPop.length < this.#popSize){
            let chosenBest= false,
                rc, count=_.rounded(spc.numToSpawn());
            while(count--){
              if(!chosenBest){
                chosenBest=true;
                baby=spc.leader.clone(this.#topology.genGID());
              }else if(spc.size() == 1 ||
                       _.rand() > Params.crossOverRate){
                baby = spc.spawn(); // no crossover
              }else{
                let [g1,g2] = spc.randPair(5);
                baby=g2 ? this.#crossOver(g1,g2) : g1.clone(this.#topology.genGID());
              }
              if(newPop.push(baby.morph()) == this.#popSize){
                break;
              }
            }
          }
        });
        return newPop;
      }
      /**
       * 1. reset appropriate values and kill off the existing phenotypes and any poorly performing species
       * 2. update and sort genomes and keep a record of the best performers
       * 3. separate the population into species of similar topology,
      */
      #cleanse(scores){
        //Resets some values ready for the next epoch, kills off all the phenotypes and any poorly performing species.
        this.#totalScoreAdj = 0;
        this.#avgScoreAdj  = 0;
        let L,tmp=[];
        this.#vecSpecies.forEach(s=>{
          if(s.stale > Params.noImprovements && s.bestScore < this.#bestScore){}else{
            tmp.push(s.purge());//keep
          }
        });
        _.append(this.#vecSpecies, tmp, true);
        //Sorts the population into descending score, keeps a record of the best n genomes and updates any score statistics accordingly.
        this.#cur.forEach((g,i)=> g.setScore(scores[i]));
        this.#cur.sort(_.comparator(_.SORT_DESC, a=>a.getScore(), b=>b.getScore()));
        this.#bestScore = Math.max(this.#bestScore,this.#cur[0].getScore());
        //save the best
        _.trunc(this.#vecBest);
        for(let i=0; i<Params.numBestElites; ++i) this.#vecBest.push(this.#cur[i]);
        /**Separates each individual into its respective species by calculating
         * a compatibility score with every other member of the population and
         * niching accordingly. The function then adjusts the score scores of
         * each individual by species age and by sharing and also determines
         * how many offspring each individual should spawn.
         */
        this.#cur.forEach((g,i)=>{
          i= this.#vecSpecies.find(s=> g.calcCompat(s.leader) <= Params.compatThreshold);
          if(i){
            i.addMember(g);
          }else{
            this.#vecSpecies.push(new Species(this.#topology, g));
          }
        });
        //now that all the genomes have been assigned a species their scores
        //need to be adjusted to take into account sharing and species age.
        this.#vecSpecies.forEach(s=> s.adjustScores())
        //calculate new adjusted total & average score for the population
        this.#totalScoreAdj= this.#cur.reduce((acc,g)=> acc + g.adjScore, this.#totalScoreAdj);
        this.#avgScoreAdj = this.#totalScoreAdj / this.#cur.length;
        //calculate how many offspring each member of the population should spawn
        this.#cur.forEach(g=> g.spawnCnt=g.adjScore / this.#avgScoreAdj);
        //calculate how many offspring each species should spawn
        this.#vecSpecies.forEach(s=> s.calcSpawnAmount());
        //so we can sort species by best score. Largest first
        this.#vecSpecies.sort(_.comparator(_.SORT_DESC, a=>a.bestScore, b=>b.bestScore));
        return this;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      NeatGA, NodeMesh, Genome, NodeGene, LinkGene, Link, Node, Species,
      ScoreFunc, InnovDB, NodeType, InnovType, RunType,
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
    gscope["io/czlab/mcfud/algo/NEAT"]=_module
  }

  //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  /**A recursive function used to calculate a lookup table of split depths.
  function _splitDepths(low, high, depth, out){
    const span = high-low;
    out.push({val: low + span/2, depth: depth+1});
    if(depth >= 4){
    }else{
      _splitDepths(low, low+span/2, depth+1, out);
      _splitDepths(low+span/2, high, depth+1, out);
    }
    return out;
  }
  split depths=4
  [{"val":0.5,"depth":1},{"val":0.25,"depth":2},{"val":0.125,"depth":3},{"val":0.0625,"depth":4},
   {"val":0.03125,"depth":5},{"val":0.09375,"depth":5},{"val":0.1875,"depth":4},{"val":0.15625,"depth":5},
   {"val":0.21875,"depth":5},{"val":0.375,"depth":3},{"val":0.3125,"depth":4},{"val":0.28125,"depth":5},
   {"val":0.34375,"depth":5},{"val":0.4375,"depth":4},{"val":0.40625,"depth":5},{"val":0.46875,"depth":5},
   {"val":0.75,"depth":2},{"val":0.625,"depth":3},{"val":0.5625,"depth":4},{"val":0.53125,"depth":5},
   {"val":0.59375,"depth":5},{"val":0.6875,"depth":4},{"val":0.65625,"depth":5},{"val":0.71875,"depth":5},
   {"val":0.875,"depth":3},{"val":0.8125,"depth":4},{"val":0.78125,"depth":5},{"val":0.84375,"depth":5},
   {"val":0.9375,"depth":4},{"val":0.90625,"depth":5},{"val":0.96875,"depth":5}]
   */


})(this)


