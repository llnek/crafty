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
 * Copyright © 2013-2022, Kenneth Leung. All rights reserved. */

;(function(gscope,UNDEF){

  "use strict";

  /**Create the module.
   */
  function _module(Core){

    if(!Core) Core=gscope["io/czlab/mcfud/core"]();
    const int=Math.floor;
    const {u:_, is}= Core;

    /**
     * @module mcfud/algo/NEAT
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
    const NeuronType={
      INPUT:0,
      HIDDEN:1,
      OUTPUT:2,
      BIAS:3,
      NONE:4
    };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**
     * @typedef {object} InnovType
     * @property {number} NEURON
     * @property {number} LINK
     */
    const InnovType={
      NEURON:0,
      LINK:1
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Select one of these types when updating the network if snapshot is chosen
     * the network depth is used to completely flush the inputs through the network.
     * active just updates the network each timestep.
     * @typedef {object} RunType
     * @property {number} SNAPSHOT
     * @property {number} ACTIVE
     */
    const RunType={
      SNAPSHOT:0,
      ACTIVE:1
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const Params={
      numInputs: 0,
      numOutputs: 0,
      bias: -1,
      //starting value for the sigmoid response
      sigmoidResponse:1,
      //number of times we try to find 2 unlinked nodes when adding a link.
      numAddLinkAttempts:5,
      //number of attempts made to choose a node that is not an input
      //node and that does not already have a recurrently looped connection to itself
      numTrysToFindLoopedLink: 5,
      //the number of attempts made to find an old link to prevent chaining in addNeuron
      numTrysToFindOldLink: 5,
      //the chance, each epoch, that a neuron or link will be added to the genome
      chanceAddLink:0.07,
      chanceAddNode:0.03,
      chanceAddRecurrentLink: -1,//0.05,
      //mutation probabilities for mutating the weights
      mutationRate:0.8,
      maxWeightPerturbation:0.5,
      probabilityWeightReplaced:0.1,
      //probabilities for mutating the activation response
      activationMutationRate:0.1,
      maxActivationPerturbation:0.1,
      //the smaller the number the more species will be created
      compatibilityThreshold:0.26,
      //during fitness adjustment this is how much the fitnesses of
      //young species are boosted (eg 1.2 is a 20% boost)
      youngFitnessBonus:1.3,
      //if the species are below this age their fitnesses are boosted
      youngBonusAgeThreshhold:10,
      //number of population to survive each epoch. (0.2 = 20%)
      survivalRate:0,
      //if the species is above this age their fitness gets penalized
      oldAgeThreshold:50,
      //by this much
      oldAgePenalty:0.7,
      crossOverRate:0.7,
      //how long we allow a species to exist without any improvement
      numGensAllowedNoImprovement:15,
      //maximum number of neurons permitted in the network
      maxPermittedNeurons:100,
      numBestElites:4
    };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Create a numeric fitness object.
     * @memberof module:mcfud/algo/NEAT
     * @param {number} v
     * @param {boolean} flipped
     * @return {object}
     */
    function NumFitness(v,flip=false){
      return{
        value:v,
        gt(b){
          return flip? this.value < b.value: this.value > b.value
        },
        eq(b){
          return this.value==b.value
        },
        lt(b){
          return flip? this.value > b.value: this.value < b.value
        },
        score(){
          return this.value
        },
        update(n){
          this.value=n;
        },
        clone(){
          return NumFitness(this.value, flip)
        }
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NeuronGene{
      /**
       * @param {NeuronType} type
       * @param {number[]} pos
       * @param {boolean} r
       */
      constructor(type, pos=null, r=false){
        this.pos= pos?pos.slice():[0,0];
        //sets curvature of sigmoid
        this.activation=1;
        this.recurrent= r;
        if(is.vec(type)){
          this.id=type[0];
          this.neuronType= type[1];
        }else{
          this.id=0;
          this.neuronType= type;
        }
      }
      clone(){
        let c= new NeuronGene(this.neuronType);
        c.id=this.id;
        c.activation=this.activation;
        c.recurrent=this.recurrent;
        c.pos=this.pos.slice();
        return c;
      }
      static from(id, type, pos=null, r=false){
        return new NeuronGene([id,type],pos,r);
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class LinkGene{
      /**
       * @param {number} from
       * @param {number} to
       * @param {number} iid
       * @param {boolean} enable
       * @param {number} w
       * @param {boolean} rec
       */
      constructor(from, to, iid, enable=true, w=null, rec = false){
        this.fromNeuron= from; //id
        this.toNeuron= to; //id
        this.innovationID= iid;
        this.recurrent= rec===true;
        this.enabled= enable !== false;
        this.weight= w===null? _.randMinus1To1() : w;
      }
      clone(){
        let c= new LinkGene();
        c.fromNeuron= this.fromNeuron;
        c.toNeuron= this.toNeuron;
        c.innovationID= this.innovationID;
        c.recurrent= this.recurrent;
        c.enabled= this.enabled;
        c.weight= this.weight;
        return c;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Innov{
      /**
       * @param {number} from
       * @param {number} to
       * @param {InnovType} t
       * @param {number} iid
       * @param {NeuronType} type
       * @param {number[]} pos
       */
      constructor(from, to, t, iid, type=NeuronType.NONE, pos=null){
        this.pos= pos?pos.slice():[0,0];
        this.innovationID= iid;
        this.innovationType=t;
        this.neuronID= 0;
        this.neuronType= type;
        this.neuronIn= from; //id
        this.neuronOut= to; //id
      }
      static from(neuron, innov_id){
        let s= new Innov(-1,-1, null, innov_id,
                         neuron.neuronType, neuron.pos);
        s.neuronID=neuron.id;
        return s;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //Given a neuron ID this function returns a clone of that neuron.
    function createNeuronFromID(history,n){
      let temp=NeuronGene.from(0,NeuronType.HIDDEN);
      for(let cur,i=0; i<history.vecInnovs.length; ++i){
        cur=history.vecInnovs[i];
        if(cur.neuronID == n){
          temp.neuronType = cur.neuronType;
          temp.id = cur.neuronID;
          temp.pos= cur.pos.slice();
          return temp;
        }
      }
      _.assert(false, "boom from createNeuronFromID");
      //return temp;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Used to keep track of all innovations created during the populations
     * evolution, adds all the appropriate innovations.
     */
    class InnovHistory{
      /**
       * @param {NeuronGene[]} neurons
       * @param {LinkGene[]} genes
       */
      constructor(neurons,genes){
        this.NEURON_COUNTER= neurons.length-1; // last known neuron id
        this.INNOV_COUNTER=0;
        this.vecInnovs= neurons.map(n=> Innov.from(n, this.nextIID())).concat(
                        genes.map(g=> new Innov(g.fromNeuron, g.toNeuron, InnovType.LINK, this.nextIID())));
      }
      nextIID(){ return this.INNOV_COUNTER++ }
      /**Checks to see if this innovation has already occurred. If it has it
       * returns the innovation ID. If not it returns a negative value.
       * @param {number} from
       * @param {number} out
       * @param {InnovType} type
       * @return {number}
       */
      check(from, out, type){
        let rc= this.vecInnovs.find(cur=> cur.neuronIn == from &&
                                          cur.neuronOut == out &&
                                          cur.innovationType == type);
        return rc===undefined?-1: rc.innovationID;
      }
      /**Creates a new innovation.
       * @param {number} from
       * @param {number} to
       * @param {InnovType} innovType
       * @param {NeuronType} neuronType
       * @param {number[]} pos
       * @return {Innov}
       */
      create(from, to, innovType, neuronType=NeuronType.NONE, pos=null){
        let i= new Innov(from, to, innovType, this.nextIID(), neuronType,  pos);
        if(InnovType.NEURON==innovType){
          i.neuronID= ++this.NEURON_COUNTER;
        }
        this.vecInnovs.push(i);
        return i;
      }
      flush(){this.vecInnovs.length=0}
      getNeuronID(inv){return this.vecInnovs[inv].neuronID}
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NLink{
      /**
       * @param {number} w
       * @param {NNeuron} from
       * @param {NNeuron} out
       * @param {boolean} rec
       */
      constructor(w, from, out, rec=false){
        this.weight=w;
        this.from=from;
        this.out=out;
        this.recurrent= rec===true;
      }
      clone(){
        let c= new NLink();
        c.weight= this.weight;
        c.from= this.from;
        c.out= this.out;
        c.recurrent= this.recurrent;
        return c;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NNeuron{
      /**
       * @param {NeuronType} type
       * @param {number} id
       * @param {number[]} pos
       * @param {number} actResponse
       */
      constructor(type, id, pos, actResponse){
        this.neuronType=type;
        this.neuronID=id;
        //sum weights*inputs
        this.sumActivation=0;
        this.output=0;
        this.posX=0;
        this.posY=0;
        this.vecLinksIn=[];
        this.vecLinksOut=[];
        this.activation=actResponse;
        this.pos= pos?pos.slice():[0,0];
      }
      clone(){
        let c= new NNeuron();
        c.neuronType=this.neuronType;
        c.neuronID= this.neuronID;
        c.output=this.output;
        c.posX=this.posX;
        c.posY=this.posY;
        c.pos= this.pos.slice();
        c.activation=this.activation;
        c.sumActivation= this.sumActivation;
        c.vecLinksIn=this.vecLinksIn.map(v=> v.clone());
        c.vecLinksOut=this.vecLinksOut.map(v=> v.clone());
        return c;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function sigmoid(netinput, response){
      return 1 / ( 1 + Math.exp(-netinput / response))
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NeuralNet{
      /**
       * @param {NNeuron[]} neurons
       * @param {number} depth
       */
      constructor(neurons, depth){
        this.vecNeurons= neurons;//own it
        this.depth=depth;
      }
      /**
       * @return {NeuralNet}
       */
      clone(){
        let c=new NeuralNet(null,this.depth);
        c.vecNeurons= this.vecNeurons.map(n=> n.clone());
        return c;
      }
      compute(inputs,type){
        return this.update(inputs, type)
      }
      /**Update network for this clock cycle.
       * @param {number[]} inputs
       * @param {RunType} type
       */
      update(inputs, type=RunType.ACTIVE){
        //if the mode is snapshot then we require all the neurons to be
        //iterated through as many times as the network is deep. If the
        //mode is set to active the method can return an output after just one iteration
        let outputs=[],
            flushCnt= type == RunType.SNAPSHOT ? this.depth:1;
        for(let sum,n,i=0; i<flushCnt; ++i){
          outputs.length=0;
          n = 0;
          //first set the outputs of the 'input' neurons to be equal
          //to the values passed into the function in inputs
          while(this.vecNeurons[n].neuronType == NeuronType.INPUT){
            this.vecNeurons[n].output = inputs[n];
            ++n;
          }
          _.assert(this.vecNeurons[n].neuronType==NeuronType.BIAS,"expecting BIAS node");
          this.vecNeurons[n].output = 1; // set bias
          //hiddens or outputs
          ++n;
          while(n < this.vecNeurons.length){
            sum= this.vecNeurons[n].vecLinksIn.reduce((acc,k)=>{ return acc + k.weight * k.from.output },0);
            //put sum thru the activation func & assign the value to this neuron's output
            this.vecNeurons[n].output = sigmoid(sum, this.vecNeurons[n].activation);
            if(this.vecNeurons[n].neuronType == NeuronType.OUTPUT){
              outputs.push(this.vecNeurons[n].output)
            }
            ++n;
          }
        }
        //the network needs to be flushed if this type of update is performed
        //otherwise it is possible for dependencies to be built on the order
        //the training data is presented
        if(type == RunType.SNAPSHOT)
          this.vecNeurons.forEach(n=> n.output=0);
        /////
        return outputs;
      }
      draw(gfx, cxLeft, cxRight, cyTop, cyBot){ }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Genome{
      /**A genome basically consists of a vector of link genes,
       * a vector of neuron genes and a fitness score.
       * @param {number} gid
       * @param {number} inputs
       * @param {number} outputs
       * @param {NeuronGene[]} neurons (optional)
       * @param {LinkGene[]} genes (optional)
       */
      constructor(gid, inputs, outputs, neurons=null, genes=null){
        if(gid<0){return}//hack for cloning
        let i,nid=0,
            inputRowSlice = 1/(inputs+2),
            outputRowSlice = 1/(outputs+1);
        if(neurons && genes){
          this.vecNeurons= neurons;//own it
          this.vecLinks= genes;
        }else{
          this.vecNeurons= [];
          this.vecLinks= [];
          for(i=0; i<inputs; ++i)
            this.vecNeurons.push(NeuronGene.from(nid++, NeuronType.INPUT, [(i+2)*inputRowSlice,0]));
          this.vecNeurons.push(NeuronGene.from(nid++, NeuronType.BIAS, [inputRowSlice,0]));
          for(i=0; i<outputs; ++i)
            this.vecNeurons.push(NeuronGene.from(nid++, NeuronType.OUTPUT, [(i+1)*outputRowSlice,1] ));
          //create the link genes, connect each input neuron to each output neuron and
          for(i=0; i<inputs+1; ++i)
            for(let j=0; j<outputs; ++j)
              this.vecLinks.push(new LinkGene(this.vecNeurons[i].id,
                                              this.vecNeurons[inputs+1+j].id, inputs+outputs+1+this.vecLinks.length));
        }
        this.nextNeuronID=nid;
        this.fitness=NumFitness(0);
        this.genomeID= gid;
        //its fitness score after it has been placed into a species and adjusted accordingly
        this.adjustedFitness=0;
        //the number of offspring is required to spawn for the next generation
        this.amountToSpawn=0;
        this.numInputs=inputs;
        this.numOutputs=outputs;
        //keeps a track of which species this genome is in (only used for display purposes)
        this.species=0;
      }
      /**Create a neural network from the genome.
       * @param {number} depth
       * @return {NeuralNet} newly created ANN
       */
      createPhenotype(depth){
        let vs= this.vecNeurons.map(n=> new NNeuron(n.neuronType,
                                                    n.id,
                                                    n.pos,
                                                    n.activation));
        this.vecLinks.forEach(k=>{
          if(k.enabled){
            let t= vs[this.getIndex(k.toNeuron) ],
                f= vs[this.getIndex(k.fromNeuron) ],
                //create a link between those two neurons
                //and assign the weight stored in the gene
                z= new NLink(k.weight, f, t, k.recurrent);
            //add new links to neuron
            f.vecLinksOut.push(z);
            t.vecLinksIn.push(z);
          }
        });
        return new NeuralNet(vs, depth);
      }
      _randAny(){
        return this.vecNeurons[_.randInt2(0, this.vecNeurons.length-1)]
      }
      _randNonInputs(){
        return this.vecNeurons[_.randInt2(this.numInputs+1, this.vecNeurons.length-1)]
      }
      /**Create a new link with the probability of Params.chanceAddLink.
       * @param {number} MutationRate
       * @param {boolean} ChanceOfLooped
       * @param {InnovHistory} history
       * @param {number} numTrysToFindLoop
       * @param {number} numTrysToAddLink
       */
      addLink(MutationRate, ChanceOfLooped, history, numTrysToFindLoop, numTrysToAddLink){
        if(_.rand() > MutationRate)
        return;
        //define holders for the two neurons to be linked. If we have find two
        //valid neurons to link these values will become >= 0.
        let nid1 = -1,
            nid2 = -1,
            recurrent = false; //flag set if a recurrent link is selected (looped or normal)
        //first test to see if an attempt shpould be made to create a
        //link that loops back into the same neuron
        if(_.rand() < ChanceOfLooped){
          while(numTrysToFindLoop--){
            let n=this._randNonInputs();
            if(n.neuronType != NeuronType.BIAS &&
               n.neuronType != NeuronType.INPUT && !n.recurrent){
              nid1 = nid2 = n.id;
              recurrent = n.recurrent = true;
              break;
            }
          }
        }else{
          while(numTrysToAddLink--){
            //choose two neurons, the second must not be an input or a bias
            nid1 = this._randAny().id;
            nid2 = this._randNonInputs().id;
            //if(nid2 == 2){ continue; }//TODO: why2?????
            if(nid1 == nid2 ||
               this.duplicateLink(nid1, nid2)){
              nid1 = nid2 = -1; // bad
            }else{
              break;
            }
          }
        }
        if(nid1 < 0 || nid2 < 0){}else{
          //console.log("addlink!!!!!");
          let id = history.check(nid1, nid2, InnovType.LINK);
          if(this.vecNeurons[this.getIndex(nid1)].pos[1]>
             this.vecNeurons[this.getIndex(nid2)].pos[1]){ recurrent = true }
          if(id<0)
            id= history.create(nid1, nid2, InnovType.LINK).innovationID;
          this.vecLinks.push(new LinkGene(nid1, nid2, id, true, _.randMinus1To1(), recurrent));
        }
      }
      /**Adds a neuron to the genotype by examining the network,
       * splitting one of the links and inserting the new neuron.
       * @param {number} MutationRate
       * @param {InnovHostory} history
       * @param {number} numTrysToFindOldLink
       */
      addNeuron(MutationRate, history, numTrysToFindOldLink){
        if(_.rand() > MutationRate)
        return;
        let chosen=0, done=false,
            fromNeuron, link1, link2, newNeuronID,
            //first a link is chosen to split. If the genome is small the code makes
            //sure one of the older links is split to ensure a chaining effect does
            //not occur. Here, if the genome contains less than 5 hidden neurons it
            //is considered to be too small to select a link at random
            SizeThreshold = this.numInputs + this.numOutputs + 5;
        if(this.vecLinks.length < SizeThreshold){
          while(numTrysToFindOldLink--){
            //choose a link with a bias towards the older links in the genome
            chosen = _.randInt2(0, this.numGenes()-1-int(Math.sqrt(this.numGenes())));
            fromNeuron = this.vecLinks[chosen].fromNeuron;
            if(this.vecLinks[chosen].enabled    &&
               !this.vecLinks[chosen].recurrent &&
               this.vecNeurons[this.getIndex(fromNeuron)].neuronType != NeuronType.BIAS){
              done = true;
              break;
            }
          }
          //failed
          if(!done){ return }
        }else{
          while(!done){
            chosen = _.randInt2(0, this.numGenes()-1);
            fromNeuron = this.vecLinks[chosen].fromNeuron;
            if(this.vecLinks[chosen].enabled &&
               !this.vecLinks[chosen].recurrent &&
               this.vecNeurons[this.getIndex(fromNeuron)].neuronType != NeuronType.BIAS){
              done = true;
            }
          }
        }
        this.vecLinks[chosen].enabled = false;
        //grab the weight from the gene (we want to use this for the weight of
        //one of the new links so that the split does not disturb anything the
        //NN may have already learned...
        let originalWeight = this.vecLinks[chosen].weight,
            from =  this.vecLinks[chosen].fromNeuron,
            to   =  this.vecLinks[chosen].toNeuron,
            //calculate the depth and width of the new neuron. We can use the depth
            //to see if the link feeds backwards or forwards
            newDepth = (this.vecNeurons[this.getIndex(from)].pos[1]+
                        this.vecNeurons[this.getIndex(to)].pos[1]) /2,
            newWidth = (this.vecNeurons[this.getIndex(from)].pos[0]+
                        this.vecNeurons[this.getIndex(to)].pos[0]) /2,
            //now see if this innovation has been created previously
            iid = history.check(from, to, InnovType.NEURON);
        /*it is possible for NEAT to repeatedly do the following:
            1. Find a link. Lets say we choose link 1 to 5
            2. Disable the link,
            3. Add a new neuron and two new links
            4. The link disabled in Step 2 maybe re-enabled when this genome
               is recombined with a genome that has that link enabled.
            5  etc etc
        Therefore, this function must check to see if a neuron ID is already
        being used. If it is then the function creates a new innovation
        for the neuron. */
        if(iid >= 0 && this.hasNeuron(history.getNeuronID(iid))) iid = -1;
        if(iid < 0){
          let new_innov= history.create(from, to,
                                        InnovType.NEURON,
                                        NeuronType.HIDDEN, [newWidth, newDepth]),
              //n= NeuronGene.from(new_innov.neuronID, NeuronType.HIDDEN, [newWidth,newDepth]);
              n= NeuronGene.from(this.nextNeuronID, NeuronType.HIDDEN, [newWidth,newDepth]);
          new_innov.neuronID=n.id;
          newNeuronID=n.id;

          this.nextNeuronID++;
          this.vecNeurons.push(n);

          //Two new link innovations are required, one for each of the
          //new links created when this gene is split.
          link1 = history.create(from, newNeuronID, InnovType.LINK).innovationID;
          this.vecLinks.push(new LinkGene(from, newNeuronID, link1, true, 1));
          link2 = history.create(newNeuronID, to, InnovType.LINK).innovationID;
          this.vecLinks.push(new LinkGene(newNeuronID, to, link2, true, originalWeight));
        }else{
          //console.log("add old neuron");
          //this innovation has already been created so grab the relevant neuron
          //and link info from the innovation database
          newNeuronID = history.getNeuronID(iid);
          link1 = history.check(from, newNeuronID, InnovType.LINK);
          link2 = history.check(newNeuronID, to, InnovType.LINK);
          //this should never happen because the innovations *should* have already occurred
          if(link1 < 0 || link2 < 0)
            _.assert(false, "Error in Genome::AddNeuron");
          //now we need to create 2 new genes to represent the new links
          this.vecLinks.push(new LinkGene(from, newNeuronID, link1, true, 1),
                             new LinkGene(newNeuronID, to, link2, true, originalWeight));
          this.vecNeurons.push(NeuronGene.from(newNeuronID, NeuronType.HIDDEN, [newWidth,newDepth]));
        }
      }
      /**Given a neuron ID, find its position in vecNeurons.
       * @param {number} neuron_id
       * @return {number}
       */
      getIndex(neuron_id){
        for(let i=0; i<this.vecNeurons.length; ++i){
          if(this.vecNeurons[i].id == neuron_id)
          return i
        }
        _.assert(false, "Error in Genome::getIndex");
      }
      /**
       * @param {number} neuronIn
       * @param {number} neuronOut
       * @return {boolean} true if the link is already part of the genome
       */
      duplicateLink(neuronIn, neuronOut){
        return this.vecLinks.some(k=> k.fromNeuron == neuronIn && k.toNeuron == neuronOut)
      }
      /**Tests to see if the parameter is equal to any existing neuron ID's.
       * @param {number} id
       * @return {boolean} true if this is the case.
       */
      hasNeuron(id){
        return this.vecNeurons.some(n=> id== n.id)
      }
      /**
       * @param {number} MutationRate
       * @param {number} ProbNewWeight the chance that a weight may get replaced by a completely new weight.
       * @param {number} maxPertubation the maximum perturbation to be applied
       */
      mutateWeights(MutationRate, ProbNewWeight, maxPertubation){
        for(let i=0; i<this.vecLinks.length; ++i){
          if(_.rand() < MutationRate){
            if(_.rand() < ProbNewWeight){
              this.vecLinks[i].weight = _.randMinus1To1() // new value
            }else{
              this.vecLinks[i].weight += _.randMinus1To1() * maxPertubation
            }
          }
        }
      }
      /**Perturbs the activation responses of the neurons.
       * @param {number} MutationRate
       * @param {number} maxPertubation the maximum perturbation to be applied
       */
      mutateActivation(MutationRate, maxPertubation){
        this.vecNeurons.forEach(n=>{
          if(_.rand() < MutationRate)
            n.activation += _.randMinus1To1() * maxPertubation
        })
      }
      /**Find the compatibility of this genome with the passed genome.
       * @param {Genome} genome
       * @return {number}
       */
      calcCompatibility(genome){
        //travel down the length of each genome counting the number of
        //disjoint genes, the number of excess genes and the number of matched genes
        let g1=0,g2=0,
            numDisjoint= 0,
            numExcess = 0,
            numMatched = 0,
            sumWeightDiff = 0;
        while((g1 < this.vecLinks.length-1) ||
              (g2 < genome.vecLinks.length-1)){
          //genome2 longer so increment the excess score
          if(g1 == this.vecLinks.length-1){ ++g2; ++numExcess; continue; }
          //genome1 longer so increment the excess score
          if(g2 == genome.vecLinks.length-1){ ++g1; ++numExcess; continue; }
          let id1 = this.vecLinks[g1].innovationID,
              id2 = genome.vecLinks[g2].innovationID;
          if(id1 == id2){
            ++g1; ++g2; ++numMatched;
            sumWeightDiff += Math.abs(this.vecLinks[g1].weight - genome.vecLinks[g2].weight);
          }else{
            ++numDisjoint;
            if(id1 < id2){ ++g1 }
            else if(id1 > id2){ ++g2 }
          }
        }
        const Disjoint = 1,
              Excess   = 1,
              Matched  = 0.4,
              longest= Math.max(this.numGenes(), genome.numGenes());
        let xxx= (Excess * numExcess/longest) + (Disjoint * numDisjoint/longest);
        return numMatched>0 ? xxx + (Matched * sumWeightDiff/ numMatched) : xxx;
      }
      sortGenes(){
        this.vecLinks.sort((a,b)=>{
          return a.innovationID < b.innovationID?-1:(a.innovationID > b.innovationID?1:0)
        });
        return this;
      }
      id(){return this.genomeID}
      setID(val){this.genomeID = val}
      numGenes(){return this.vecLinks.length}
      numNeurons(){return this.vecNeurons.length}
      setFitness(num){this.fitness = NumFitness(num)}
      splitY(val){return this.vecNeurons[val].pos[1]}
      genes(){return this.vecLinks}
      neurons(){return this.vecNeurons}
      startOfGenes(){return 0}
      endOfGenes(){return this.vecLinks.length}
      clone(){
        let src=this,
            c=new Genome(-911);
        c.fitness= src.fitness.clone();
        c.genomeID= src.genomeID;
        c.adjustedFitness= src.adjustedFitness;
        c.amountToSpawn= src.amountToSpawn;
        c.numInputs= src.numInputs;
        c.numOutputs= src.numOutputs;
        c.species= src.species;
        c.vecNeurons= src.vecNeurons.map(v=> v.clone());
        c.vecLinks= src.vecLinks.map(v=> v.clone());
        return c;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Species{
      /**
       * @param {Genome} firstOrg
       */
      constructor(sid, firstOrg){
        this.speciesID= sid;
        //generations since fitness has improved, we can use
        //this info to kill off a species if required
        this._gensNoImprovement=0;
        //age of species
        this._age=0;
        //how many of this species should be spawned for the next population
        this.spawnsRqd=0;
        this.vecMembers= [firstOrg];
        this._leader= firstOrg.clone();
        //best fitness found so far by this species
        this._bestFitness= firstOrg.fitness.score();
      }
      /**Adjusts the fitness of each individual by first
       * examining the species age and penalising if old, boosting if young.
       * Then we perform fitness sharing by dividing the fitness
       * by the number of individuals in the species.
       * This ensures a species does not grow too large.
       */
      adjustFitnesses(){
        let score,total = 0;
        this.vecMembers.forEach(m=>{
          score = m.fitness.score();
          if(this._age < Params.youngBonusAgeThreshhold){
            //boost the fitness scores if the species is young
            score *= Params.youngFitnessBonus
          }
          if(this._age > Params.oldAgeThreshold){
            //punish older species
            score *= Params.oldAgePenalty
          }
          total += score;
          //apply fitness sharing to adjusted fitnesses
          m.adjustedFitness = score/this.vecMembers.length;
        })
      }
      /**Adds a new member to this species and updates the member variables accordingly
       * @param {Genome} newMember
       */
      addMember(newMember){
        if(newMember.fitness.score() > this._bestFitness){
          this._bestFitness = newMember.fitness.score();
          this._gensNoImprovement = 0;
          this._leader = newMember.clone();
        }
        this.vecMembers.push(newMember);
        newMember.species= this.id();
      }
      /**Clears out all the members from the last generation, updates the age and gens no improvement.
       */
      purge(){
        this.vecMembers.length=0;
        ++this._gensNoImprovement;
        this.spawnsRqd = 0;
        ++this._age;
        return this;
      }
      /**Simply adds up the expected spawn amount for each individual
       * in the species to calculate the amount of offspring
       * this species should spawn.
       */
      calculateSpawnAmount(){
        this.vecMembers.forEach(m=>{
          this.spawnsRqd += m.amountToSpawn
        })
      }
      /**Spawns an individual from the species selected at random
       * from the best Params::dSurvivalRate percent.
       * @return {Genome} a random genome selected from the best individuals
       */
      spawn(){
        let n,baby;
        if(this.vecMembers.length == 1){
          baby = this.vecMembers[0]
        }else{
          n = int(Params.survivalRate * this.vecMembers.length)-1;
          if(n<0)n=1;
          baby = this.vecMembers[ _.randInt2(0, n) ];
        }
        return baby.clone();
      }
      id(){return this.speciesID}
      bestFitness(){return this._bestFitness }
      age(){return this._age}
      leader(){return this._leader}
      numToSpawn(){return this.spawnsRqd}
      numMembers(){return this.vecMembers.length}
      gensNoImprovement(){return this._gensNoImprovement}
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function sortSpecies(s){
      return s.sort((lhs,rhs)=>{
        //so we can sort species by best fitness. Largest first
        return lhs._bestFitness > rhs._bestFitness?-1:(
          lhs._bestFitness < rhs._bestFitness?1:0
        )
      })
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**A recursive function used to calculate a lookup table of split depths.
     */
    function split(low, high, depth, out){
      const span = high-low;
      out.push({val: low + span/2, depth: depth+1});
      if(depth > 6){
      }else{
        split(low, low+span/2, depth+1, out);
        split(low+span/2, high, depth+1, out);
      }
      return out;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NeatGA{
      /**Creates a base genome from supplied values and creates a population
       * of 'size' similar (same topology, varying weights) genomes.
       * @param {number} size
       * @param {number} inputs
       * @param {number} outputs
       */
      constructor(size, inputs, outputs){
        let dummy= new Genome(0,inputs, outputs);

        //this holds the precalculated split depths. They are used
        //to calculate a neurons x/y position for rendering and also
        //for calculating the flush depth of the network when a
        //phenotype is working in 'snapshot' mode.
        //create the network depth lookup table
        this.vecSplits = split(0, 1, 0, []);
        this.vecBestGenomes=[];
        this.vecSpecies=[];

        this.SPECIES_COUNTER=0;
        this.GENOME_COUNTER=0;

        this.generation=0;
        this.popSize=size;
        //adjusted fitness scores
        this.totFitAdj=0;
        this.avFitAdj=0;
        //index into the genomes for the fittest genome
        this.fittestGenome=0;
        this._bestEverFitness=0;
        this.vecGenomes= _.fill(size, ()=> new Genome(this.nextGID(), inputs, outputs));
        //create the innovation list. First create a minimal genome
        this.innovHistory= new InnovHistory(dummy.neurons(), dummy.genes());
      }
      /**Resets some values ready for the next epoch, kills off
       * all the phenotypes and any poorly performing species.
       */
      resetAndKill(){
        this.totFitAdj = 0;
        this.avFitAdj  = 0;
        let L,tmp=[];
        this.vecSpecies.forEach(s=>{
          if(s.gensNoImprovement() > Params.numGensAllowedNoImprovement &&
             s.bestFitness() < this._bestEverFitness){
            //kill,delete it
          }else{
            //keep it
            tmp.push(s.purge());
          }
        });
        this.vecSpecies.length=0;
        tmp.forEach(t=> this.vecSpecies.push(t));
      }
      /**Separates each individual into its respective species by calculating
       * a compatibility score with every other member of the population and
       * niching accordingly. The function then adjusts the fitness scores of
       * each individual by species age and by sharing and also determines
       * how many offspring each individual should spawn.
       */
      speciateAndCalculateSpawnLevels(){
        let added = false;
        this.vecGenomes.forEach(g=>{
          for(let s=0; s<this.vecSpecies.length; ++s){
            if(g.calcCompatibility(this.vecSpecies[s].leader()) <= Params.compatibilityThreshold){
              this.vecSpecies[s].addMember(g);
              added = true;
              break;
            }
          }
          if(!added)
            this.vecSpecies.push(new Species(this.nextSID(), g));
          added = false;
        });
        //now all the genomes have been assigned a species the fitness scores
        //need to be adjusted to take into account sharing and species age.
        this.vecSpecies.forEach(s=> s.adjustFitnesses())
        //calculate new adjusted total & average fitness for the population
        this.vecGenomes.forEach(g=> this.totFitAdj += g.adjustedFitness);
        //////
        this.avFitAdj = this.totFitAdj/this.vecGenomes.length;
        //calculate how many offspring each member of the population should spawn
        this.vecGenomes.forEach(g=> g.amountToSpawn=g.adjustedFitness / this.avFitAdj);
        //calculate how many offspring each species should spawn
        this.vecSpecies.forEach(s=> s.calculateSpawnAmount());
      }
      /**
       * @param {Genome} mum
       * @param {Genome} dad
       */
      crossOver(mum, dad){
        const MUM=0, DAD=1;
        //first, calculate the genome we will using the disjoint/excess
        //genes from. This is the fittest genome.
        let best;
        //if they are of equal fitness use the shorter (because we want to keep
        //the networks as small as possible)
        if(mum.fitness.score() == dad.fitness.score()){
          best=mum.numGenes() == dad.numGenes() ? (_.randSign()>0?DAD:MUM)
                                                : (mum.numGenes() < dad.numGenes()?MUM :DAD)
        }else{
          best = mum.fitness.score() > dad.fitness.score() ? MUM : DAD
        }
        function addNeuronID(nid, vec){ if(vec.indexOf(nid)<0) vec.push(nid) }
        //these vectors will hold the offspring's nodes and genes
        let babyNeurons=[],
            babyGenes=[],
            vecNeurons=[],
            //create iterators so we can step through each parents genes and set
            //them to the first gene of each parent
            //this will hold a copy of the gene we wish to add at each step
            curMum=0,curDad=0,selectedGene;
        //step through each parents genes until we reach the end of both
        while(!(curMum == mum.endOfGenes() && curDad == dad.endOfGenes())){
          if(curMum == mum.endOfGenes()&&curDad != dad.endOfGenes()){
            //the end of mum's genes have been reached
            if(best == DAD) selectedGene = dad.vecLinks[curDad];
            ++curDad;
          }else if(curDad == dad.endOfGenes() && curMum != mum.endOfGenes()){
            //the end of dads's genes have been reached
            if(best == MUM) selectedGene = mum.vecLinks[curMum];
            ++curMum;
          }else if(mum.vecLinks[curMum].innovationID < dad.vecLinks[curDad].innovationID){
            if(best == MUM) selectedGene = mum.vecLinks[curMum];
            ++curMum;
          }else if(dad.vecLinks[curDad].innovationID < mum.vecLinks[curMum].innovationID){
            if(best == DAD) selectedGene = dad.vecLinks[curDad];
            ++curDad;
          }else if(dad.vecLinks[curDad].innovationID == mum.vecLinks[curMum].innovationID){
            selectedGene=_.rand() < 0.5 ? mum.vecLinks[curMum] : dad.vecLinks[curDad];
            ++curMum;
            ++curDad;
          }
          //add the selected gene if not already added
          if(babyGenes.length == 0 ||
             _.last(babyGenes).innovationID != selectedGene.innovationID){
            babyGenes.push(selectedGene.clone())
          }
          //Check if we already have the nodes referred to in SelectedGene.
          //If not, they need to be added.
          addNeuronID(selectedGene.fromNeuron, vecNeurons);
          addNeuronID(selectedGene.toNeuron, vecNeurons);
        }
        //now create the required nodes
        vecNeurons.sort().forEach(n=> babyNeurons.push(createNeuronFromID(this.innovHistory,n)));
        /////
        return new Genome(this.nextGID(), mum.numInputs, mum.numOutputs, babyNeurons, babyGenes);
      }
      nextSID(){ return ++this.SPECIES_COUNTER}
      nextGID(){ return ++this.GENOME_COUNTER }
      /**
       * @param {number} numComparisons
       * @return {Genome}
       */
      tournamentSelection(comparisons){
        let chosen = 0,
            bestSoFar = 0;
        //Select NumComparisons members from the population at random testing
        //against the best found so far
        for(let g,i=0, z=this.vecGenomes.length-1; i<comparisons; ++i){
          g = _.randInt2(0, z);
          if(this.vecGenomes[g].fitness.score() > bestSoFar){
            chosen = g;
            bestSoFar = this.vecGenomes[g].fitness.score();
          }
        }
        return this.vecGenomes[chosen];
      }
      /**Searches the lookup table for the splitY value of each node
       * in the genome and returns the depth of the network based on this figure.
       * @param {Genome} gen
       * @return {number}
       */
      calculateNetDepth(gen){
        let maxSoFar = 0;
        for(let nd=0; nd<gen.numNeurons(); ++nd){
          for(let i=0; i<this.vecSplits.length; ++i)
            if(gen.splitY(nd) == this.vecSplits[i].val &&
               this.vecSplits[i].depth > maxSoFar){
              maxSoFar = this.vecSplits[i].depth;
            }
        }
        return maxSoFar + 2;
      }
      /**Sorts the population into descending fitness, keeps a record of the best
       * n genomes and updates any fitness statistics accordingly.
       */
      sortAndRecord(scores){
        this.vecGenomes.forEach((g,i)=> g.setFitness(scores[i]));
        this.vecGenomes.sort((a,b)=>{
          return a.fitness.score()>b.fitness.score()?-1:(a.fitness.score()<b.fitness.score()?1:0)
        });
        this._bestEverFitness = Math.max(this._bestEverFitness,this.vecGenomes[0].fitness.score());
        //save the best
        this.vecBestGenomes.length=0;
        for(let i=0; i<Params.numBestElites; ++i)
          this.vecBestGenomes.push(this.vecGenomes[i]);
      }
      /**Performs one epoch of the genetic algorithm and returns a vector of pointers to the new phenotypes.
       * @param {number[]} fitnessScores
       * @return {}
       */
      epoch(scores){
        _.assert(scores.length == this.vecGenomes.length, "NeatGA::Epoch(scores/ genomes mismatch)!");
        //reset appropriate values and kill off the existing phenotypes and any poorly performing species
        this.resetAndKill();
        //update and sort genomes and keep a record of the best performers
        this.sortAndRecord(scores);
        //separate the population into species of similar topology,
        this.speciateAndCalculateSpawnLevels();
        //this will hold the new population of genomes
        let baby2,baby,newPop=[],numSpawnedSoFar = 0;
        this.vecSpecies.forEach(spc=>{
          if(numSpawnedSoFar<this.popSize){
            let chosenBest= false,
                numToSpawn = _.rounded(spc.numToSpawn());
            while(numToSpawn--){
              if(!chosenBest){
                chosenBest=true;
                baby = spc.leader().clone();
              }else{
                if(spc.numMembers() == 1){
                  baby = spc.spawn()
                }else{
                  let numAttempts = 5,
                      g2,g1 = spc.spawn();
                  if(_.rand() < Params.crossOverRate){
                    g2 = spc.spawn();
                    while(g1.id() == g2.id() && (numAttempts--)){
                      g2 = spc.spawn()
                    }
                    baby= g1.id() != g2.id() ? this.crossOver(g1, g2) : g1;
                  }else{
                    baby = g1
                  }
                }
                baby.setID(this.nextGID());
                //now mutate this baby
                if(baby.numNeurons() < Params.maxPermittedNeurons)
                  baby.addNeuron(Params.chanceAddNode,
                                 this.innovHistory, Params.numTrysToFindOldLink);
                //now there's the chance a link may be added
                baby.addLink(Params.chanceAddLink,
                             Params.chanceAddRecurrentLink,
                             this.innovHistory,
                             Params.numTrysToFindLoopedLink, Params.numAddLinkAttempts);
                //mutate the weights
                baby.mutateWeights(Params.mutationRate,
                                   Params.probabilityWeightReplaced,
                                   Params.maxWeightPerturbation);
                baby.mutateActivation(Params.activationMutationRate,
                                      Params.maxActivationPerturbation);
              }
              newPop.push(baby.sortGenes());
              if(++numSpawnedSoFar == this.popSize){ numToSpawn = 0 }
            }
          }
        });
        //if there is an underflow due to the rounding error and the amount
        //of offspring falls short of the population size additional children
        //need to be created and added to the new population. This is achieved
        //simply, by using tournament selection over the entire population.
        if(numSpawnedSoFar < this.popSize){
          //calculate amount of additional children required
          let rqd = this.popSize - numSpawnedSoFar;
          while(rqd--)
            newPop.push(this.tournamentSelection(int(this.popSize/ 5)).clone());
        }
        //replace the current population with the new one
        this.vecGenomes = newPop;
        ++this.generation;
        return this.createPhenotypes();
      }
      /**Cycles through all the members of the population and creates their phenotypes.
       * @return {NeuralNet[]} the new phenotypes
       */
      createPhenotypes(){
        return this.vecGenomes.map(g=> g.createPhenotype( this.calculateNetDepth(g)))
      }
      //renders the best performing species statistics and a visual aid
      //showing the distribution.
      renderSpeciesInfo(){}
      numSpecies(){return this.vecSpecies.length}
      bestEverFitness(){return this._bestEverFitness}
      /**
       * @return {NeuralNet[]} the n best phenotypes from the previous generation.
       */
      getBestPhenotypesFromLastGeneration(){
        return this.vecBestGenomes.map((g,i)=> {
          g.createPhenotype(this.calculateNetDepth(g))
        })
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      NeatGA, NeuralNet, Genome, NeuronGene, LinkGene, NLink, NNeuron, Species,
      NumFitness, InnovHistory, NeuronType, InnovType, RunType,
      configParams(options){
        return _.inject(Params,options)
      }
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("../main/core"))
  }else{
    gscope["io/czlab/mcfud/algo/NEAT"]=_module
  }

})(this)


