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



