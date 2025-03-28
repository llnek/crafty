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

			get numNeurons(){return this.#numNeurons}

			/**
			 * @param {number} numNeurons
			 * @param {number} numInputsPerNeuron
			 * @param {function} actFunc
			 * @param {boolean} wantBiasNodes
			 */
			constructor(numNeurons, numInputsPerNeuron, actFunc, wantBiasNode=true){
				this.#neurons= _.fill(numNeurons,()=> new Neuron(this,numInputsPerNeuron,wantBiasNode));
				this.#actFunc=actFunc;
				this.#numNeurons=numNeurons;
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

			/**
			 * @param {number} inputs
			 * @param {number} outputs
			 * @param {any[]} hidden
			 * @param {function} actFuncOut
			 * @param {boolean} wantBiasNode
			 */
			constructor(inputs, outputs, [numHidden,perHidden,actFunc], actFuncOut, wantBiasNode=true){
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

			#scoreCalcTarget;
			#scoreCalc;
			#genes;
			#age;

			get age(){return this.#age}
			set age(a){ this.#age=a; }

			/**
			 * @param {any[]} genes
			 * @param {func} scoreCalculator
			 * @param {object} target
			 */
			constructor(genes, scoreCalculator, target){
				this.#scoreCalc=scoreCalculator;
				this.#scoreCalcTarget= target;
				this.#genes=genes;
				this.#age=0;
			}
			/**
			 * @return {array}
			 */
			getScoreCalcInfo(){
				return [this.#scoreCalc, this.#scoreCalcTarget]
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
			 * @return {any[]} copy of our genes
			 */
			copyGenes(){
				return this.#genes.slice()
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
				this.updateScore(this.#scoreCalcTarget ?
					                 this.#scoreCalc.call(this.#scoreCalcTarget, this.#genes) : this.#scoreCalc(this.#genes));
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
				return new ChromoNumero(this._genes(), f, t);
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

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		const _$={

			NeuronLayer,
			Neuron,
			NeuralNet,

			ChromoNumero,
			Chromosome,

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


