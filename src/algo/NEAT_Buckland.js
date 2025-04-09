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
      sigmoid: function(x){
        return 1 / ( 1 + Math.exp(-x));
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
              obj.output = Params.sigmoid(obj.funcOverInLinks(_sum)/ obj.activation);
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


