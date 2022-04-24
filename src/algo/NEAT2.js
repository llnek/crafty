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
     * @module mcfud/algo/NEAT2
     */

    const NeuronType={
      INPUT:0, OUTPUT:1,BIAS:2,HIDDEN:3
    };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Create a numeric fitness object.
     * @memberof module:mcfud/algo/NEAT2
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
        update(v){
          this.value=v;
        },
        clone(){
          return NumFitness(this.value, flip)
        }
      }
    }

    const NumFIT=NumFitness;
    const Params={
      BIAS:1,
      nextInnovNo: 1,
      superSpeed:1,
      mutationRate:0.8,
      crossOverRate:0.25,
      probAddLink:0.07,
      probAddNode:0.03,
      probCancelLink:0.75,
      probAddRecurrentLink: 0.05,
      probWeightReplaced:0.1,
      maxWeightPerturbation:1/50,
      //how long we allow a species to exist without any improvement
      noImprovementLimit:15,
      excessCoeff: 1,
      weightDiffCoeff: 0.5,
      compatibilityThreshold : 3 //0.26
    };

    /**A connection between 2 nodes
     * @class
     */
    class LinkGene{
      /**
       * @param {Neuron} from
       * @param {Neuron} to
       * @param {number} w
       * @param {number} innovID
       */
      constructor(from, to, w, innovID){
        this.innovNo = innovID;
        this.fromNode = from;
        this.toNode = to;
        this.weight = w;
        this.enabled = true;
      }
      /**
       */
      mutateWeight(){
        if(_.rand()<Params.probWeightReplaced){
          this.weight = _.randMinus1To1();
        }else{
          this.weight += _.randGaussian() * Params.maxWeightPerturbation;
        }
        if(this.weight>1){ this.weight=1 }
        if(this.weight < -1){ this.weight = -1 }
      }
      clone(from, to){
        let c= new LinkGene(from, to, this.weight, this.innovNo);
        c.enabled = this.enabled;
        return c;
      }
    }

    /**Genetic history.
     * @class
     */
    class InnovHistory{
      /**
       * @param {number} from
       * @param {number} to
       * @param {number} inno
       * @param {number[]} innovationNos
       */
      constructor(from, to, inno, innovationNos){
        this.fromNode = from;
        this.toNode = to;
        this.innovNumber = inno;
        //the innovation numbers from the connections of the genome which first had this mutation
        //this represents the genome and allows us to test if another genoeme is the same
        //this is before this connection was added
        this.innovNumbers = innovationNos.slice();
      }
      /**Check if the genome matches the original genome and
       * the connection is between the same neurons.
       * @param {Genome} genome
       * @param {Neuron} from
       * @param {Neuron} to
       * @return {boolean}
       */
      matches(genome, from, to){
        //if the number of connections are different then the genoemes aren't the same
        if(genome.genes.length == this.innovNumbers.length &&
           from.number == this.fromNode && to.number == this.toNode){
          //check if all the innovation numbers match from the genome
          for(let i=0; i<genome.genes.length; ++i){
            if(!this.innovNumbers.includes(genome.genes[i].innovNo)){
              return false
            }
          }
          //if reached this far then the innovationNumbers match the genes
          //and the connection is between the same nodes so it does match
          return true;
        }
        return false;
      }
    }

    /**
     * @class
     */
    class Neuron{
      /**
       * @param {NeuronType} type
       * @param {number} id
       * @param {number} layer
       */
      constructor(type, id, layer=0){
        this.outputConnections = []; //LinkGenes
        this.layer = layer;
        this.neuronType=type;
        this.pos = [0,0];
        this.number = id;
        this.inputSum = 0; //current sum i.e. before activation
        this.outputValue = 0; //after activation function is applied
      }
      /**The neuron sends its output to the inputs
       * of the neurons its connected to.
       * @return {Neuron} this
       */
      engage(){
        //no sigmoid for the inputs and bias
        if(this.layer != 0)
          this.outputValue = this.sigmoid(this.inputSum);
        for(let i=0; i<this.outputConnections.length; ++i){
          //sum += (input[i] * weight[i])
          if(this.outputConnections[i].enabled)
            this.outputConnections[i].toNode.inputSum += this.outputConnections[i].weight * this.outputValue;
        }
        return this;
      }
      /**Activation function.
       * @return {number} 1 or 0
       */
      stepFunction(x){
        return x<0?0:1
      }
      /**Activation function.
       * @return {number}
       */
      sigmoid(x){
        //y=1/(1+e^(-x))
        return 1 / (1 + Math.pow(Math.E, -4.9*x))
      }
      sigmoid2(netinput, response){
        return 1 / ( 1 + Math.exp(-netinput / response))
      }
      /**Whether this neuron is connected to the `node`.
       * @param {Neuron} node
       * @return {boolean}
       */
      isConnectedTo(node){
        let a,b;
        if(node.layer<this.layer){
          a=node;b=this;
        }else if(node.layer>this.layer){
          a=this; b=node;
        }
        if(a && b)
          for(let i=0; i<a.outputConnections.length; ++i){
            if(a.outputConnections[i].toNode === b) return true;
          }
        return false;
      }
      clone(){
        return new Neuron(this.neuronType, this.number,this.layer)
      }
    }

    /**
     * @class
     */
    class Genome{
      /**
       * @param {number} inputs
       * @param {number} outputs
       * @param {boolean} crossOver
       */
      constructor(inputs, outputs, crossOver=false){
        this.outputs = outputs;
        this.fitness=NumFIT(0);
        this.inputs = inputs;
        this.layers = 2;
        this.nextNode = 0;
        //the phenotype
        this.network = [];
        //genes=== connections
        this.genes = [];
        //nodes=== neurons
        this.nodes = crossOver? [] : this.ctor(inputs,outputs,[]);
      }
      ctor(inputs,outputs,nodes){
        //make the inputs nodes
        for(let i=0; i<inputs; ++i){
          nodes.push(new Neuron(NeuronType.INPUT, i));
          ++this.nextNode;
        }
        //and the output nodes
        for(let i=0; i<outputs; ++i){
          nodes.push(new Neuron(NeuronType.OUTPUT, i+inputs, 1));
          ++this.nextNode;
        }
        //lock onto the BIAS node
        this.biasNode = this.nextNode;
        this.nextNode++;
        nodes.push(new Neuron(NeuronType.BIAS, this.biasNode));
        return nodes;
      }
      /**
       * @param {object} x
       * @return {Genome} this
       */
      bindTo(x){
        this.husk=x;
        return (x.brain=this);
      }
      /**Get the neuron with a matching number.
       * @param {number} id
       * @return {Neuron}
       */
      getNeuron(id){
        return this.nodes.find(n=> n.number == id)
      }
      /**Build up the network.
       * @return {Genome} this
       */
      connectNeurons(){
        //clean slate
        this.nodes.forEach(n=> n.outputConnections.length=0);
        //then
        this.genes.forEach(g=> g.fromNode.outputConnections.push(g));
        return this;
      }
      /**Feeding in input values to the NN and returning output array
       * @param {number[]} values
       * @return {number[]}
       */
      update(values){
        _.assert(this.network.length>0,"invalid network");
        _.assert(values.length==this.inputs, "invalid input values");
        for(let n,i=0; i<this.inputs; ++i){
          n=this.nodes[i];
          n.outputValue = values[i];
          _.assert(n.neuronType==NeuronType.INPUT, "invalid input neuron");
        }

        //run it
        this.nodes[this.biasNode].outputValue = Params.BIAS;
        this.network.forEach(n=> n.engage());

        //the outputs are this.nodes[inputs] to this.nodes [inputs+outputs-1]
        const outs=[];
        for(let n,i=0; i<this.outputs; ++i){
          n=this.nodes[this.inputs + i];
          outs[i] = n.outputValue;
          _.assert(n.neuronType==NeuronType.OUTPUT, "invalid output neuron");
        }

        //reset all the this.nodes for the next feed forward
        this.nodes.forEach(n=> n.inputSum = 0);

        return outs;
      }
      compute(values){
        return this.update(values)
      }
      /**Sets up the NN.
       * @return {Genome} this
       */
      generateNetwork(){
        this.connectNeurons();
        _.cls(this.network);
        //for each layer add the node in that layer,
        //since layers cannot connect to themselves
        //there is no need to order nodes within a layer
        for(let y=0; y<this.layers; ++y)
          for(let i=0; i<this.nodes.length; ++i)
            if(this.nodes[i].layer == y) this.network.push(this.nodes[i]);
        return this;
      }
      /**Mutate the NN by adding a new node, it does this by
       * picking a random connection and disabling it then 2
       * new connections are added 1 between the input node of
       * the disabled connection and the new node and the other
       * between the new node and the output of the disabled connection
       * @param {InnovHistory[]} history
       * @return {Genome} this
       */
      addNeuron(history){
        if(this.genes.length == 0)
          return this.addLink(history);
        //////
        let tries=10,rc =0;
        if(this.genes.length>1)
          rc=_.randInt(this.genes.length);
        //don't want BIAS node
        while(this.genes[rc].fromNode.neuronType==NeuronType.BIAS && tries>0){
          rc=_.randInt(this.genes.length);
          tries--;
        }

        if(tries<=0){
          console.warn("failed to add neuron");
          return this;
        }

        this.genes[rc].enabled = false;
        ///
        let cinv, newNode,
            newNodeNo=this.nextNode;
        this.nodes.push(new Neuron(NeuronType.HIDDEN,newNodeNo));
        ++this.nextNode;
        //add a new link to the new node with a weight = 1
        newNode=this.getNeuron(newNodeNo);
        cinv = this.getInnovNo(history, this.genes[rc].fromNode, newNode);
        this.genes.push(new LinkGene(this.genes[rc].fromNode, newNode, 1, cinv));
        cinv = this.getInnovNo(history, newNode, this.genes[rc].toNode);
        //add a new link from the new node with a weight the same as the old connection
        this.genes.push(new LinkGene(newNode, this.genes[rc].toNode, this.genes[rc].weight, cinv));
        newNode.layer = 1+this.genes[rc].fromNode.layer;
        ///
        cinv = this.getInnovNo(history, this.nodes[this.biasNode], newNodeNo);
        //connect the bias to the new node with a weight of 0
        this.genes.push(new LinkGene(this.nodes[this.biasNode], newNode, 0, cinv));
        //if the layer of the new node is equal to the layer of the output node of the old connection then a new layer needs to be created
        //more accurately the layer numbers of all layers equal to or greater than this new node need to be incrimented
        if(newNode.layer == this.genes[rc].toNode.layer){
          //dont include this newest node
          for(let i=0; i<this.nodes.length-1; ++i){
            if(this.nodes[i].layer >= newNode.layer){
              this.nodes[i].layer+=1;
            }
          }
          ++this.layers;
        }
        return this.connectNeurons();
      }
      /**
       * @param {InnovHistory[]} history
       * @return {Genome} this
       */
      addLink(history){
        if(this.fullyConnected()){
          console.warn("addLink failed, too full");
          return this;
        }
        let rn1 = _.randInt(this.nodes.length),
            rn2 = _.randInt(this.nodes.length),
            tries=10, cin,temp,badPair=(r1,r2)=>{
              return this.nodes[r1].layer == this.nodes[r2].layer ||
                     this.nodes[r1].isConnectedTo(this.nodes[r2])
            };
        while(badPair(rn1, rn2) && tries>0){
          rn1 = _.randInt(this.nodes.length);
          rn2 = _.randInt(this.nodes.length);
          --tries;
        }

        if(tries<=0){
          console.log("failed to add-link");
          return this;
        }

        if(this.nodes[rn1].layer > this.nodes[rn2].layer){
          temp = rn2;
          rn2 = rn1;
          rn1 = temp;
        }
        //get the innovation number of the connection
        //this will be a new number if no identical genome has mutated in the same way
        cin = this.getInnovNo(history, this.nodes[rn1], this.nodes[rn2]);
        //add the connection with a random array
        //changed this so if error here
        this.genes.push(new LinkGene(this.nodes[rn1], this.nodes[rn2], _.randMinus1To1(), cin));
        return this.connectNeurons();
      }
      /**Get the innovation number for the new mutation
       * if this mutation has never been seen before then
       * it will be given a new unique innovation number
       * if this mutation matches a previous mutation then
       * it will be given the same innovation number as the previous one
       * @param {InnovHistory[]} history
       * @param {Neuron} from
       * @param {Neuron} to
       * @return {number}
       */
      getInnovNo(history, from, to){
        let isNew = true,
            innovNo = Params.nextInnovNo;
        for(let i=0; i<history.length; ++i){
          if(history[i].matches(this, from, to)){
            isNew = false;
            //set the innovation number as the innovation number of the match
            innovNo = history[i].innovNumber;
            break;
          }
        }
        //if the mutation is new then create an arrayList of varegers representing the current state of the genome
        if(isNew){
          //then add this mutation to the innovationHistory
          history.push(new InnovHistory(from.number, to.number,
                                        innovNo,
                                        this.genes.map(g=> g.innovNo)));
          Params.nextInnovNo+=1;
        }
        return innovNo;
      }
      /**Check whether the network is fully connected or not.
       * @return {boolean}
       */
      fullyConnected(){
        let maxConnections = 0,
            nodesInLayers = _.fill(this.layers, 0);
        this.nodes.forEach(n=> nodesInLayers[n.layer] += 1);
        //for each layer the maximum amount of connections is the number in this layer * the number of this.nodes infront of it
        //so lets add the max for each layer together and then we will get the maximum amount of connections in the network
        for(let before, i=0; i<this.layers-1; ++i){
          before = 0;
          for(let j= i+1; j<this.layers; ++j){
            before += nodesInLayers[j];
          }
          maxConnections += before*nodesInLayers[i];
        }
        //if the number of connections is equal to the max number of connections possible then it is full
        return maxConnections <= this.genes.length;
      }
      /**
       * @param {InnovHistory[]} history
       * @return {Genome} this
       */
      mutate(history){
        if(this.genes.length == 0)
          this.addLink(history);

        if(_.rand() < Params.mutationRate){
          this.genes.forEach(g=> g.mutateWeight())
        }

        if(_.rand() < Params.probAddLink){
          this.addLink(history)
        }

        //1% of the time add a node
        if(_.rand() < Params.probAddNode){
          this.addNeuron(history);
        }

        return this;
      }
      /**Called when this Genome is better that the other parent
       * @param {Genome} parent2
       * @return {Genome} new child
       */
      crossOver(parent2){
        let isEnabled = [],  // bools
            childGenes = [], // LinkGene
            child = new Genome(this.inputs, this.outputs, true);
        //copy of this
        child.nextNode = this.nextNode;
        child.layers = this.layers;
        child.biasNode = this.biasNode;
        //all inherited genes
        let p2,setEnabled;
        for(let i=0; i<this.genes.length; ++i){
          setEnabled = true;
          p2 = this.matchingGene(parent2, this.genes[i].innovNo);
          if(p2 != -1){
            if(!this.genes[i].enabled || !parent2.genes[p2].enabled){
              if(_.rand() < Params.probCancelLink) setEnabled = false;
            }
            childGenes.push(_.rand() < 0.5 ? this.genes[i] : parent2.genes[p2]);
          }else{
            //disjoint or excess gene
            childGenes.push(this.genes[i]);
            setEnabled = this.genes[i].enabled;
          }
          isEnabled.push(setEnabled);
        }
        //since all excess and disjovar genes are inherrited from
        //the more fit parent (this Genome) the childs structure is
        //no different from this parent | with exception of dormant
        //connections being enabled but this wont effect this.nodes
        //so all the this.nodes can be inherrited from this parent
        this.nodes.forEach(n=> child.nodes.push(n.clone()));
        //clone all the connections
        for(let i=0; i< childGenes.length; ++i){
          child.genes.push(childGenes[i].clone(child.getNeuron(childGenes[i].fromNode.number),
                                               child.getNeuron(childGenes[i].toNode.number)));
          child.genes[i].enabled = isEnabled[i];
        }
        return child.connectNeurons();
      }
      /**Check whether or not there is a gene matching
       * the input innovation number  in the input genome.
       * @param {Genome} p
       * @param {number} innov
       * @return {number}
       */
      matchingGene(p, innov){
        for(let i=0; i<p.genes.length; ++i){
          if(p.genes[i].innovNo == innov) return i;
        }
        return -1;
      }
      /**Prints out info about the genome to the console
       * @return {Genome} this
       */
      printGenome(){
        console.log("Prvar genome  layers:" + this.layers);
        console.log("bias node: " + this.biasNode);
        console.log("this.nodes");
        for(let i=0; i<this.nodes.length; ++i){
          console.log(this.nodes[i].number + ",");
        }
        console.log("Genes");
        for(let i=0; i<this.genes.length; ++i){
          console.log("gene " + this.genes[i].innovNo +
                      "From node " + this.genes[i].fromNode.number +
                      "To node " + this.genes[i].toNode.number +
                      "is enabled " + this.genes[i].enabled +
                      "from layer " + this.genes[i].fromNode.layer +
                      "to layer " + this.genes[i].toNode.layer + "weight: " + this.genes[i].weight);
        }
        return this;
      }
      /**Make a copy.
       * @return {Genome}  new guy
       */
      clone(){
        let c= new Genome(this.inputs, this.outputs, true);
        this.nodes.forEach(n=> c.nodes.push(n.clone()));
        //copy all the connections so that they connect the clone new this.nodes
        this.genes.forEach(g=>{
          c.genes.push(g.clone(c.getNeuron(g.fromNode.number),
                               c.getNeuron(g.toNode.number)))
        });
        c.fitness=this.fitness.clone();
        c.layers = this.layers;
        c.nextNode = this.nextNode;
        c.biasNode = this.biasNode;
        return c.connectNeurons();
      }
      /**Draw the genome on the screen
       * @return {Genome} this
       */
      drawGenome(startX, startY, w, h){
        return this;
      }
    }

    /**
     * @class
     */
    class Species{
      /**
       * @param {Genome} p
       */
      constructor(p){
        this.bestFitness = 0;
        this.members = [];
        this.rep;
        this.staleness = 0; //how many generations the species has gone without an improvement
        this.averageFitness = 0;
        if(p){
          //since it is the only one in the species it is by default the best
          this.bestFitness = p.fitness.score();
          this.members.push(p);
          this.rep = p.clone();//champion
        }
      }
      /**Check whether the parameter genome is in this species.
       * @param {Genome} g
       * @return {boolean}
       */
      compatible(g){
        let excessAndDisjoint = this.getExcessDisjoint(g, this.rep),
            averageWeightDiff = this.averageWeightDiff(g, this.rep),
            compatibility, largeGenomeNormaliser = g.genes.length - 20;
        if(largeGenomeNormaliser < 1){
          largeGenomeNormaliser = 1
        }
        //compatibility formula
        compatibility = (Params.excessCoeff * excessAndDisjoint / largeGenomeNormaliser) +
                        (Params.weightDiffCoeff * averageWeightDiff);
        return Params.compatibilityThreshold > compatibility;
      }
      /**
       * @param {Genome} p
       * @return {Species} this
       */
      add(p){
        this.members.push(p);
        return this;
      }
      /**Get the number of excess and disjoint genes between
       * the 2 input genomes i.e. returns the number of genes which dont match.
       * @param {Genome} brain1
       * @param {Genome} brain2
       * @return {number}
       */
      getExcessDisjoint(brain1, brain2){
        let matching = 0;
        for(let i=0; i< brain1.genes.length; ++i){
          for(let j=0; j< brain2.genes.length; ++j){
            if(brain1.genes[i].innovNo == brain2.genes[j].innovNo){
              ++matching;
              break;
            }
          }
        }
        //return no of excess and disjoint genes
        return (brain1.genes.length + brain2.genes.length - 2*matching);
      }
      /**Get the avereage weight difference between matching genes in the input genomes
       * @param {Genome} brain1
       * @param {Genome} brain2
       * @return {number}
       */
      averageWeightDiff(brain1, brain2){
        if(brain1.genes.length == 0 ||
           brain2.genes.length == 0) { return 0 }
        let matching = 0,
            totalDiff = 0;
        for(let i=0; i< brain1.genes.length; ++i){
          for(let j=0; j< brain2.genes.length; ++j){
            if(brain1.genes[i].innovNo == brain2.genes[j].innovNo){
              ++matching;
              totalDiff += Math.abs(brain1.genes[i].weight - brain2.genes[j].weight);
              break;
            }
          }
        }
        return matching == 0? 100: totalDiff / matching;
      }
      /**Sort via fitness.
       */
      sortAsc(){
        this.members.sort((a,b)=>{
          return a.fitness.score()>b.fitness.score()?-1:( a.fitness.score()<b.fitness.score()?1:0)
        });
        if(this.members.length == 0){
          this.staleness = 200
        }else if(this.members[0].fitness.score() > this.bestFitness){
          this.bestFitness = this.members[0].fitness.score();
          this.rep = this.members[0].clone();
          this.staleness = 0;
        }else{
          ++this.staleness
        }
      }
      /**
       */
      setAverage(){
        this.averageFitness = this.members.reduce((acc,p)=>{ return acc + p.fitness.score() },0) / this.members.length;
      }
      /**Gets baby from the this.players in this species
       * @param {InnovHistory[]} history
       * @return {Genome} new guy
       */
      giveMeBaby(history){
        let baby, select=()=>{
          let total = this.members.reduce((acc,p)=>{
            return acc + p.fitness.score()
          },0),
              sum = 0, slice = _.rand() * total;
          for(let i=0; i< this.members.length; ++i){
            sum += this.members[i].fitness.score();
            if(sum > slice)
              return this.members[i];
          }
          return this.members[0];
        }
        if(_.rand() < Params.crossOverRate){
          baby = select().clone();
        }else{
          let par1 = select(),
              par2 = select();
          //the crossover function expects the highest fitness parent
          //to be the object and the lowest as the argument
          baby= par1.fitness.score() < par2.fitness.score() ? par2.crossOver(par1) : par1.crossOver(par2);
        }
        return baby.mutate(history);
      }
      /**Kills off bottom half of the species
       */
      cull(){
        if(this.members.length > 2)
          this.members.length= int(this.members.length / 2);
      }
      /**In order to protect unique this.players,
       * the fitnesses of each player is divided by
       * the number of this.players in the species that that player belongs to
       */
      fitnessSharing(){
        this.members.forEach(p=>{
          p.fitness.update(p.fitness.score()/this.members.length)
        })
      }
    }

    /**
     * @class
     */
    class NeatGA{
      /**
       * @param {number} size
       * @param {number} inputs
       * @param {number} outputs
       */
      constructor(size,inputs,outputs){
        this.history = [];
        this.species = [];
        this.gen = 1;
        this.genomes = _.fill(size, (i,g)=>{
          g=new Genome(inputs, outputs);
          g.mutate(this.history);
          return g.generateNetwork();
        });
      }
      /**Cycles through all the members of the population and creates their phenotypes.
       * @return {Genome[]} the current phenotypes
       */
      createPhenotypes(){
        return this.genomes;
      }
      /**Called when all the players are dead and a new generation needs to be made.
       */
      epoch(scores){
        let prevBest = this.gen==1? null: this.genomes[0];
        this.speciate();
        this.genomes.forEach((g,i)=>{ if(i>=0) g.fitness.update(scores[i]) });
        this.sortSpecies();
        this.resetAndKill();
        let children = [],
            numToSpawn,averageSum = this.getAvgFitnessSum();
        this.species.forEach(s=>{
          //add champion without any mutation
          children.push(s.rep.clone());
          //the number of children species is allowed,
          //note -1 is because the champ is already added
          numToSpawn = int(s.averageFitness/averageSum * this.genomes.length)-1;
          for(let i=0; i< numToSpawn; ++i)
            children.push(s.giveMeBaby(this.history));
        });
        if(!prevBest)
          prevBest=this.species[0].rep;
        if(children.length < this.genomes.length){
          children.push(prevBest.clone())
        }
        //get babies from the best species
        while(children.length < this.genomes.length)
          children.push(this.species[0].giveMeBaby(this.history));
        _.append(this.genomes, children, true);
        this.gen +=1;
        this.genomes.forEach(g=> g.generateNetwork());
        return this.genomes;
      }
      /**Seperate genomes into species based on how similar they are
       * to the leaders of each species in the previous run
       */
      speciate(){
        this.species.forEach(s=> _.cls(s.members));
        for(let f,g,i=0; i< this.genomes.length; ++i){
          g=this.genomes[i];
          f= false;
          for(let s=0;s<this.species.length;++s){
            if(this.species[s].compatible(g)){
              this.species[s].add(g);
              f= true;
              break;
            }
          }
          if(!f)
            this.species.push(new Species(g));
        }
      }
      /**Sorts the players within a specie and the species by their fitnesses
       */
      sortSpecies(){
        //sort each species internally
        this.species.forEach(s=> s.sortAsc());
        //then sort species by the fitness of its best player
        this.species.sort((a,b)=>{
          return a.bestFitness>b.bestFitness?-1:( a.bestFitness<b.bestFitness?1:0)
        });
      }
      /**Get the sum of each this.species average fitness
       * @return {number}
       */
      getAvgFitnessSum(){
        return this.species.reduce((acc,s)=>{
          return acc + s.averageFitness
        },0);
      }
      /**
       */
      resetAndKill(){
        let temp=[];
        this.species.forEach(s=>{
          s.cull(); //kill bottom half
          s.fitnessSharing(); //also while we're at it lets do fitness sharing
          s.setAverage(); //reset averages because they will have changed
        });
        //rid of stale ones
        for(let i=0; i<this.species.length; ++i){
          if(i<2){
            temp.push(this.species[i]);
          }else if(this.species[i].staleness < Params.noImprovementLimit){
            temp.push(this.species[i]); //keep it
          }
        }
        _.append(this.species, temp, true);
        //rid of bad
        let avgSum = this.getAvgFitnessSum();
        temp=[];
        for(let i=0; i<this.species.length; ++i){
          if(i<1){
            temp.push(this.species[i])
          }else if((this.species[i].averageFitness / avgSum * this.species.length)< 1){
            //delete
          }else{
            temp.push(this.species[i]); //keep
          }
        }
        _.append(this.species,temp,true);
      }
      massExtinction(){
        if(this.species.length>5) this.species.slice(0,5);
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      NeatGA, Genome, LinkGene, Neuron, Species,
      NumFitness, InnovHistory,
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
    gscope["io/czlab/mcfud/algo/NEAT2"]=_module
  }

})(this)


