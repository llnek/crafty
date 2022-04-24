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
     * @module mcfud/algo/NNetGA
     */

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		const Params={

			mutationRate: 0.1,
			crossOverRate: 0.7,
			probTournament: 0.75,

      NUM_HIDDEN: 1,
      BIAS:-1,
      NUM_ELITE:4,
      TOURNAMENT_SIZE :5,
      MAX_PERTURBATION: 0.3,
      ACTIVATION_RESPONSE: 1,
      NEURONS_PER_HIDDEN: 10

    };

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**Fitness Interface.
		 * @class
		 */
		class Fitness{
			/**
			 */
			constructor(){}
			/**
			 * @param {Fitness} x
			 * @return {boolean}
			 */
			gt(x){}
			/**
			 * @param {Fitness} x
			 * @return {boolean}
			 */
			lt(x){}
			/**
			 * @param {Fitness} x
			 * @return {boolean}
			 */
			eq(x){}
			/**
			 * @return {Fitness}
			 */
			clone(){}
			/**
			 * @return {number}
			 */
			score(){}
			/**
			 * @param {any} n
			 */
			update(n){}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**Numeric fitness.
		 * @class
		 */
		class NumFitness extends Fitness{
			/**
			 * @param {number} v
			 * @param {boolean} flip default false
			 */
			constructor(v,flip){
				super();
				this.value=v;
				this.flip=flip;
			}
			/**
			 * @param {NumFitness} b
			 * @return {boolean}
			 */
			gt(b){
				return this.flip? this.value < b.value: this.value > b.value
			}
			/**
			 * @param {NumFitness} b
			 * @return {boolean}
			 */
			eq(b){
				return this.value==b.value
			}
			/**
			 * @param {NumFitness} b
			 * @return {boolean}
			 */
			lt(b){
				return this.flip? this.value > b.value: this.value < b.value
			}
			/**
			 * @return {number}
			 */
			score(){
				return this.value
			}
			/**
			 * @param {number} n
			 */
			update(n){
				this.value=n
			}
			/**
			 * @return {NumFitness}
			 */
			clone(){
				return new NumFitness(this.value, this.flip);
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @property {number} averageScore
		 * @property {number} totalScore
		 * @property {number} bestScore
		 * @property {number} worstScore
		 * @property {object} best
		 */
		class Statistics{
			/**
			 */
			constructor(){
				this.averageScore=0;
				this.totalScore=0;
				this.bestScore=0;
				this.worstScore=0;
				this.best=UNDEF;
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @property {number} numInputs
		 * @property {number} activation
		 * @property {number} error
		 * @property {number[]} weights
		 */
		class Neuron{
			/**
			 * @param {number} inputs
			 */
			constructor(inputs){
				//we need an additional weight for the bias hence the +1
				const ws= _.fill(inputs+1, ()=> _.randMinus1To1());
				this.numInputs= ws.length;
				this.activation=0;
				this.weights=ws;
				this.error=0;
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @property {number} numNeurons
		 * @property {Neuron[]} neurons
		 */
		class NeuronLayer{
			/**
			 * @param {number} numNeurons
			 * @param {number} numInputsPerNeuron
			 */
			constructor(numNeurons, numInputsPerNeuron){
				this.numNeurons=numNeurons;
				this.neurons= _.fill(numNeurons,()=> new Neuron(numInputsPerNeuron));
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @class
		 */
		class NeuralNet{
			/**
			 * @param {number} inputs
			 * @param {number} outputs
			 * @param {number} numHidden
			 * @param {number} neuronsPerHidden
			 */
			constructor(inputs, outputs, numHidden, neuronsPerHidden){
				//create the layers of the network
				this.layers=(function(out){
					if(numHidden>0){
						out.push(new NeuronLayer(neuronsPerHidden, inputs));
						for(let i=0; i<numHidden-1; ++i)
							out.push(new NeuronLayer(neuronsPerHidden,neuronsPerHidden));
					}
					return _.conj(out,new NeuronLayer(outputs, numHidden>0?neuronsPerHidden:inputs));
				})([]);

				this.numOfWeights=this.layers.reduce((sum,y)=>{
					return sum + y.neurons.reduce((acc,u)=>{
						return acc+u.weights.length
					},0)
				},0);

				this.numOutputs=outputs;
				this.numInputs=inputs;
				this.numHidden=numHidden;
				this.neuronsPerHidden=neuronsPerHidden;
			}
			/**
			 * @param {number[]} weights
			 */
			putWeights(weights){
				_.assert(weights.length>=this.numOfWeights,"bad input to putWeights");
				let pos=0;
				this.layers.forEach(y=> y.neurons.forEach(u=> u.weights.forEach((v,i)=> u.weights[i]= weights[pos++])))
			}
			/**
			 * @return {number[]}
			 */
			getWeights(){
				const out=[];
				for(let i=0; i<this.numHidden+1; ++i)
					for(let j=0; j<this.layers[i].numNeurons; ++j)
						for(let k=0; k<this.layers[i].neurons[j].numInputs; ++k){
							out.push(this.layers[i].neurons[j].weights[k])
						}
				return out;
			}
			/**
			 * @return {number}
			 */
			getNumberOfWeights(){
				return this.numOfWeights
			}
			/**Same as update.
			 * @param {number[]}
			 * @return {number[]}
			 */
			feedForward(inputs){
				return this.update(inputs)
			}
			/**
			 * @param {number[]} inputs
			 * @return {number[]}
			 */
			update(inputs){
				_.assert(inputs.length >= this.numInputs,"invalid input size");
				let sum,nodes,idx, out=[];
				this.layers.forEach((y,i)=>{
					if(i>0)
						inputs=out;
					out=[];
					y.neurons.forEach(u=>{
						idx=0;
						sum=0;
						nodes=u.numInputs;
						//skip the bias hence -1
						for(let k=0;k<nodes-1;++k)
							sum += (u.weights[k] * inputs[idx++]);
						sum += (u.weights[nodes-1] * Params.BIAS);
						out.push(u.activation= this.sigmoid(sum, Params.ACTIVATION_RESPONSE));
					});
				});
				return _.assert(out.length== this.numOutputs, "out length incorrect") ? out : [];
			}
			/**
			 * @private
			 */
			sigmoid(input, response){
				return 1 / (1 + Math.exp(-input / response))
			}
			/**
			 * @return {number[]}
			 */
			calcSplitPoints(){
				let pts= [],
						pos = 0;
				this.layers.forEach(y=> y.neurons.forEach(u=>{
					pos += u.numInputs;
					pts.push(pos-1);
				}));
				return pts;
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		/**
		 * @property {number} age
		 * @property {any[]} genes
		 * @property {Fitness} fitness
		 */
		class Chromosome{
			/**
			 * @param {any[]} genes
			 * @param {Fitness} fitness
			 */
			constructor(genes, fitness){
				this.fitness=fitness;
				this.genes=genes;
				this.age=0;
			}
			/**
			 * @return {Chromosome}
			 */
			clone(){
				return new Chromosome(this.genes.slice(),this.fitness.clone())
			}
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function randSpan(genes){
			let a= _.randInt(genes.length),
					b= _.randInt(genes.length);
			return a<b ? [a,b] : [b,a];
		}

		/**Choose two random points and “scramble” the genes located between them.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {any[]} genes
		 */
		function mutateSM(genes){
			if(_.rand() < Params.mutationRate){
				let [beg, end] = randSpan(genes);
				let tmp,count= end-beg-1;
				if(count==2){
					tmp=genes[beg+1];
					genes[beg+1]=genes[beg+2];
					genes[beg+2]=tmp;
				}else if(count>2){
					tmp=_.shuffle(genes.slice(beg+1,end));
					for(let k=0,i=beg+1;i<end;++i){
						genes[i]=tmp[k++]
					}
				}
			}
		}

		/**Select two random points, grab the chunk of chromosome
		 * between them and then insert it back into the chromosome
		 * in a random position displaced from the original.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {any[]} genes
		 */
		function mutateDM(genes){
			if(_.rand() < Params.mutationRate){
				let [beg, end]= randSpan(genes);
				let p,tmp,rem,
						N=genes.length,count= end-beg-1;
				if(count>0){
					tmp=genes.slice(beg+1,end);
					rem=genes.slice(0,beg+1).concat(genes.slice(end));
					p=_.randInt(rem.length);
					tmp=rem.slice(0,p).concat(tmp).concat(rem.slice(p));
					genes.length=0;
					tmp.forEach(v=> genes.push(v));
					_.assert(genes.length==N,"mutateDM error");
				}
			}
		}

		/**Almost the same as the DM operator, except here only one gene is selected
		 * to be displaced and inserted back into the chromosome.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {any[]} genes
		 */
		function mutateIM(genes){
			if(_.rand() < Params.mutationRate){
				//choose a gene to move
				let pos=_.randInt(genes.length),
						left,right,N=genes.length,v = genes[pos];
				//remove from the chromosome
				genes.splice(pos,1);
				//move the iterator to the insertion location
				pos = _.randInt(genes.length);
				left=genes.slice(0,pos);
				right=genes.slice(pos);
				genes.length=0;
				left.forEach(n=> genes.push(n));
				genes.push(v);
				right.forEach(n=> genes.push(n));
				_.assert(N==genes.length,"mutateIM error");
			}
		}

		/**Select two random points and reverse the genes between them.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {any[]} genes
		 */
		function mutateIVM(genes){
			if(_.rand()<Params.mutationRate){
				let [beg, end]= randSpan(genes);
				let tmp,N=genes.length,count= end-beg-1;
				if(count>1){
					tmp=genes.slice(beg+1,end).reverse();
					for(let k=0, i=beg+1;i<end;++i){
						genes[i]=tmp[k++];
					}
				}
				_.assert(N==genes.length,"mutateIVM error");
			}
		}

		/**Select two random points, reverse the order between the two points,
		 * and then displace them somewhere along the length of the original chromosome.
		 * This is similar to performing IVM and then DM using the same start and end points.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {any[]} genes
		 */
		function mutateDIVM(genes){
			if(_.rand()<Params.mutationRate){
				let [beg, end]= randSpan(genes);
				let N=genes.length,
						p,tmp,rem,count= end-beg-1;
				if(count>0){
					tmp=genes.slice(beg+1,end).reverse();
					rem=genes.slice(0,beg+1).concat(genes.slice(end));
					p=_.randInt(rem.length);
					tmp=rem.slice(0,p).concat(tmp).concat(rem.slice(p));
					genes.length=0;
					tmp.forEach(v=> genes.push(v));
					_.assert(genes.length==N,"mutateDIVM error");
				}
			}
		}

		/**Several genes are chosen at random from one parent and
		 * then the order of those selections is imposed on
		 * the respective genes in the other parent.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {any[]} mum
		 * @param {any[]} dad
		 * @return {array}
		 */
		function crossOverOBX(mum,dad){
			let temp, positions,
					b1,b2,cpos, pos = _.randInt2(0, mum.length-2);
			b1 = mum.slice();
			b2 = dad.slice();
			if(_.rand() > Params.crossOverRate || mum === dad){}else{
				positions=_.listIndexesOf(mum,true).slice(0, _.toGoldenRatio(mum.length)[1]).sort();
				temp=positions.map(p=> mum[p]);
				//so now we have n amount of genes from mum in the temp
				//we can impose their order in dad.
				cpos = 0;
				for(let cit=0; cit<b2.length; ++cit){
					for(let i=0; i<temp.length; ++i){
						if(b2[cit]==temp[i]){
						 b2[cit] = temp[cpos++];
						 break;
						}
					}
				}
				//now vice versa
				temp.length=0;
				cpos = 0;
				//first grab from the same positions in dad
				for(let i=0; i<positions.length; ++i){
					temp.push(dad[positions[i]])
				}
				//and impose their order in mum
				for(let cit=0; cit<b1.length; ++cit){
					for(let i=0; i<temp.length; ++i){
						if(b1[cit]==temp[i]){
							b1[cit] = temp[cpos++];
							break;
						}
					}
				}
			}
			return [b1, b2];
		}

		/**Similar to Order-Based CrossOver, but instead of imposing the order of the genes,
		 * this imposes the position.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} mum
		 * @param {array} dad
		 * @return {array}
		 */
		function crossOverPBX(mum, dad){
			let b1,b2;
			if(_.rand() > Params.crossOverRate || mum === dad){
				b1 = mum.slice();
				b2 = dad.slice();
			}else{
				//initialize the babies with null values so we can tell which positions
				//have been filled later in the algorithm
				b1=_.fill(mum.length, null);
				b2=_.fill(mum.length, null);
				let positions=_.listIndexesOf(mum,true).slice(0, _.toGoldenRatio(mum.length)[1]).sort();
				//now we have chosen some cities it's time to copy the selected cities
				//over into the offspring in the same position.
				positions.forEach(i=>{
					b1[i] = mum[i];
					b2[i] = dad[i];
				});
				//fill in the blanks. First create two position markers so we know
				//whereabouts we are in b1 and b2
				let c1=0, c2=0;
				for(let i=0; i<mum.length; ++i){
					//advance position marker until we reach a free position in b2
					while(b2[c2] !==null && c2 < mum.length){ ++c2 }
					//b2 gets the next from mum which is not already present
					if(b2.indexOf(mum[i])<0){
						b2[c2] = mum[i]
					}
					//now do the same for baby1
					while(b1[c1] !==null && c1 < mum.length){ ++c1 }
					//b1 gets the next from dad which is not already present
					if(b1.indexOf(dad[i])<0){
						b1[c1] = dad[i]
					}
				}
				_.assert(!b1.some(x=> x===null), "crossOverPBX null error");
				_.assert(!b2.some(x=> x===null), "crossOverPBX null error");
			}
			return [b1,b2];
		}

		/**
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} mum
		 * @param {array} dad
		 * @return {array}
		 */
		function crossOverRND(mum,dad){
			let b1,b2;
			if(_.rand() > Params.crossOverRate || mum===dad){
				b1 = mum.slice();
				b2 = dad.slice();
			}else{
				let cp = _.randInt(mum.length);
				b1=[];
				b2=[];
				for(let i=0; i<cp; ++i){
					b1.push(mum[i]);
					b2.push(dad[i]);
				}
				for(let i=cp; i<mum.length; ++i){
					b1.push(dad[i]);
					b2.push(mum[i]);
				}
			}
			return [b1,b2];
		}

		/**Partially matched crossover.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {any[]} mum
		 * @param {any[]} dad
		 * @return {array}
		 */
		function crossOverPMX(mum, dad){
			let b1 = mum.slice(),
					b2 = dad.slice();
			if(_.rand() > Params.crossOverRate || mum === dad){}else{
				//first we choose a section of the chromosome
				let beg = _.randInt2(0, mum.length-2);
				let end = beg;
				while(end <= beg)
					end = _.randInt2(0, mum.length-1);
				//now we iterate through the matched pairs of genes from beg
				//to end swapping the places in each child
				for(let p1,p2,g1,g2,pos=beg; pos<end+1; ++pos){
					//these are the genes we want to swap
					g1 = mum[pos];
					g2 = dad[pos];
					if(g1 != g2){
						//find and swap them in b1
						p1 = b1.indexOf(g1);
						p2 = b1.indexOf(g2);
						_.swap(b1, p1,p2);
						//and in b2
						p1 = b2.indexOf(g1);
						p2 = b2.indexOf(g2);
						_.swap(b2, p1,p2);
					}
				}
			}
			return [b1,b2];
		}

		/**
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} mum
		 * @param {array} dad
		 * @param {array} splitPoints
		 * @return {array}
		 */
		function crossOverAtSplits(mum, dad, splitPoints){
			let b1, b2;
			if(_.rand() > Params.crossOverRate || mum === dad){
				b1=mum.slice();
				b2=dad.slice();
			}else{
				//determine two crossover points
				let cp1 = splitPoints[_.randInt2(0, splitPoints.length-2)],
						cp2 = splitPoints[_.randInt2(cp1, splitPoints.length-1)];
				b1=[];
				b2=[];
				//create the offspring
				for(let i=0; i<mum.length; ++i){
					if(i<cp1 || i>=cp2){
						//keep the same genes if outside of crossover points
						b1.push(mum[i]);
						b2.push(dad[i]);
					}else{
						//switch over the belly block
						b1.push(dad[i]);
						b2.push(mum[i]);
					}
				}
			}
			return [b1,b2];
		}

		/**Roulette selection.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} pop
		 * @param {number} totalScore
		 * @return {Chromosome}
		 */
		function getChromoRoulette(pop, totalScore){
			let hit, sum = 0, slice = _.rand() * totalScore;
			for(let i=0; i<pop.length; ++i){
				sum += pop[i].fitness.score();
				//if the fitness so far > random number return the chromo at
				//this point
				if(sum >= slice){
					hit = pop[i];
					break;
				}
			}
			return hit;
		}

		/**Roulette selection with probabilities.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} pop
		 * @param {number} totalScore
		 * @return {Chromosome}
		 */
		function chromoRoulette(pop,totalScore){
			let i,prev=0,R=_.rand();
			let ps=pop.map(p=>{ return prev= (prev+ p.fitness.score()/totalScore) });
			for(i=0;i<ps.length-1;++i)
				if(R >= ps[i] && R <= ps[i+1]) return pop[i]
			return pop[0];
		}

		/**
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} pop
		 * @param {number} N
		 * @return {Chromosome}
		 */
		function tournamentSelectionN(pop,N){
			let chosenOne = 0,
					bestSoFar = NumFitness(-Infinity);
			//Select N members from the population at random testing against
			//the best found so far
			for(let i=0; i<N; ++i){
				let thisTry = _.randInt(pop.length);
				if(pop[thisTry].fitness.gt(bestSoFar)){
					chosenOne = thisTry;
					bestSoFar = pop[thisTry].fitness;
				}
			}
			return pop[chosenOne];
		}

		/**
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} pop current generation
		 * @return {Chromosome}
		 */
		function tournamentSelection(pop){
			let g1 = _.randInt(pop.length),
					g2 = _.randInt(pop.length);
			//make sure they are different
			while(g1 == g2)
				g2 = _.randInt2(0,pop.length-1);
			if(_.rand() < Params.probTournament){
				return pop[g1].fitness.gt(pop[g2].fitness)? pop[g1] : pop[g2]
			}else{
				return pop[g1].fitness.lt(pop[g2].fitness)? pop[g1] : pop[g2]
			}
		}

		/**Calculate statistics on population based on scores.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} pop current generation
		 * @param {boolean} flip true if smaller score is better
		 * @return {Statistics}
		 */
		function calcStats(pop,flip){
			let best= 0,
					worst= Infinity,
					stats=new Statistics();
			if(flip){ worst=0; best=Infinity; }
			function B(c){
				best = c.fitness.score();
				stats.bestScore = best;
				stats.best= c;
			}
			function W(c){
				worst = c.fitness.score();
				stats.worstScore = worst;
			}
			pop.forEach(c=>{
				if(flip){
					if(c.fitness.score() < best){
						B(c)
					}else if(c.fitness.score() > worst){
						W(c)
					}
				}else{
					if(c.fitness.score() > best){
						B(c)
					}else if(c.fitness.score() < worst){
						W(c)
					}
				}
				stats.totalScore += c.fitness.score();
			});
			stats.averageScore = stats.totalScore / pop.length;
			return stats;
		}

		/**This type of fitness scaling sorts the population into ascending
		 * order of fitness and then simply assigns a fitness score based on
		 * its position in the ladder.
		 * (so if a genome ends up last it gets score of zero,
		 * if best then it gets a score equal to the size of the population.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} pop current generation
		 * @return {Statistics}
		 */
		function fitnessScaleRank(pop){
			//sort population into ascending order
			pop.sort((a,b)=>{
				return a.fitness.lt(b.fitness)?-1:(
					a.fitness.gt(b.fitness)?1:0
				)
			});
			//now assign fitness according to the genome's position on
			//this new fitness 'ladder'
			pop.forEach((p,i)=> p.fitness.update(i));
			//recalculate values used in selection
			return calcStats(pop);
		}

		/**Scales the fitness using sigma scaling.
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {array} pop current generation
		 * @param {Statistics} stats
		 * @return {array} [sigma, new_stats]
		 */
		function fitnessScaleSigma(pop, stats){
			let i,total = 0;
			//first iterate through the population to calculate the standard deviation
			for(i=0; i<pop.length; ++i){
				total += (pop[i].fitness.score() - stats.averageScore) *
								 (pop[i].fitness.score() - stats.averageScore);
			}
			let old,variance = total/pop.length;
			//standard deviation is the square root of the variance
			let sigma = Math.sqrt(variance);
			//now iterate through the population to reassign the fitness scores
			pop.forEach(p=>{
				old= p.fitness.score();
				p.fitness.update((old-stats.averageScore)/(2*sigma));
			});
			return [sigma, calcStats(pop)];
		}

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
		function fitnessScaleBoltzmann(pop, boltzmannTemp){
			//reduce the temp a little each generation
			boltzmannTemp -= Parmas.BOLTZMANN_DT;
			//make sure it doesn't fall below minimum value
			if(boltzmannTemp< Parmas.MIN_TEMP) boltzmannTemp = Parmas.MIN_TEMP;
			//iterate through the population to find the average e^(fitness/temp)
			//keep a record of e^(fitness/temp) for each individual
			let expBoltz=[],
					i,average = 0;
			pop.forEach((p,i)=>{
				expBoltz.push(Math.exp(p.fitness.score() / boltzmannTemp));
				average += expBoltz[i];
			});
			average /= pop.length;
			//now iterate once more to calculate the new expected values
			pop.forEach((p,i)=> p.fitness.update(expBoltz[i]/average));
			//recalculate values used in selection
			return [boltzmannTemp, calcStats(pop)];
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function markStart(extra,fld="cycles"){
			let s= extra.startTime=_.now();
			extra[fld]=0;
			return s;
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function markEnd(extra){
			return extra.endTime=_.now();
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function newChild(p1, parents, crossOver, mutate, calcFit){
      let p2= _.randInt(parents.length);
			while(parents.length>1 && p2==p1){
				p2= _.randInt(parents.length)
			}
			let c1=parents[p1].genes,
					c,b1,b2,c2=parents[p2].genes;
			if(crossOver){
				[b1,b2]=crossOver(c1,c2);
			}else{
				b1=c1.slice();
				b2=c2.slice();
			}
      if(mutate){
        mutate(b1);
				mutate(b2);
      }
			let f1= calcFit(b1, parents[p1].fitness),
					f2= calcFit(b2, parents[p2].fitness);
			return f1.gt(f2)? new Chromosome(b1, f1): new Chromosome(b2, f2);
    }

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function bisectLeft(arr,e){
			//ascending array
      let a,i=0;
      for(;i<arr.length;++i){
        a=arr[i];
        if(a.fitness.eq(e.fitness) ||
           e.fitness.lt(a.fitness)) break;
      }
      return i;
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function* getNextStar([start,maxMillis],{
			mutate,create,maxAge,
			calcFit,poolSize,crossOver
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
        if(par.fitness.gt(bestPar.fitness)){
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
        child = newChild(pindex, parents, crossOver, mutate, calcFit);
        if(par.fitness.gt(child.fitness)){
          if(maxAge===undefined){ continue }
          par.age += 1;
					if(maxAge > par.age){ continue }
          index = bisectLeft(history, child, 0, history.length);
          ratio= index / history.length;
          if(_.rand() < Math.exp(-ratio)){
            parents[pindex] = child;
            continue;
          }
          bestPar.age = 0;
          parents[pindex] = bestPar;
          continue;
        }
        if(!child.fitness.gt(par.fitness)){
          //same fitness
          child.age = par.age + 1;
          parents[pindex] = child;
          continue;
        }
				//child is better, so replace the parent
				child.age = 0;
				parents[pindex] = child;
				//replace best too?
        if(child.fitness.gt(bestPar.fitness)){
          yield (bestPar = child);
          history.push(bestPar);
				}
      }
    }

		/**
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {Fitness} optimal
		 * @param {object} extra
		 * @return {array}
		 */
		function runGASearch(optimal,extra){
			let start= markStart(extra),
					maxCycles=(extra.maxCycles|| 100),
					maxMillis= (extra.maxSeconds || 30) * 1000,
					imp, now, gen= getNextStar([start,maxMillis],extra);
			while(true){
				imp= gen.next().value;
				now= markEnd(extra);
				if(now-start > maxMillis){
					now=null;
					break;
				}
				if(!optimal.gt(imp.fitness)){
					break;
				}
				if(extra.cycles >= maxCycles){
					break;
				}
				extra.cycles += 1;
				//console.log(imp.genes.join(","));
			}
			return [now==null, imp]
		}

		/**
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {number|array} pop
		 * @param {object} extra
		 * @return {array}
		 */
		function runGACycle(pop,extra){
			let {maxCycles, targetScore, maxSeconds}=extra;
			let s,now, start= markStart(extra),
          maxMillis= (maxSeconds || 30) * 1000;
			maxCycles= maxCycles || 100;
			while(true){
				pop= genPop(pop, extra);
				now= markEnd(extra);
				//time out?
				if(now-start > maxMillis){
					now=null;
					break;
				}
				//pop.forEach(p=> console.log(p.genes.join("")));
				s=calcStats(pop);
				//matched?
				if(_.echt(targetScore) &&
					 s.bestScore >= targetScore){ break }
				//too many?
				if(extra.cycles>= maxCycles){ break }
				extra.cycles += 1;
			}
			extra.gen++;
			return [now == null, pop];
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function genPop(pop,{
			calcFit, crossOver, create,mutate
		})
		{
			if(is.num(pop))
				return _.fill(pop, ()=> create());

			let b1,b2,res,mum,dad,vecNewPop = [];
			let stats=calcStats(pop);

			//ascending
			pop.sort((a,b)=> a.fitness.lt(b.fitness)?-1:(a.fitness.gt(b.fitness)?1:0));
			for(let k=Params.NUM_ELITES, i=pop.length-1;i>=0;--i){
				if(k>0){
					vecNewPop.push(pop[i]);
					--k;
				}else{
					break;
				}
			}

			while(vecNewPop.length < pop.length){
				if(Params.TOURNAMENT_SIZE !== undefined){
					mum = tournamentSelection(pop,Params.TOURNAMENT_SIZE);
					dad = tournamentSelection(pop,Params.TOURNAMENT_SIZE);
				}else{
					mum = chromoRoulette(pop,stats);
					dad = chromoRoulette(pop,stats);
				}
				if(crossOver){
					[b1,b2]= crossOver(mum.genes,dad.genes);
				}else{
					b1=mum.genes.slice();
					b2=dad.genes.slice();
				}
				if(mutate){
					mutate(b1);
					mutate(b2);
				}
				vecNewPop.push(new Chromosome(b1, calcFit(b1, mum.fitness)),
											 new Chromosome(b2, calcFit(b2,dad.fitness)));
			}

			while(vecNewPop.length > pop.length){
				vecNewPop.pop();
			}

			return vecNewPop;
		}

		/**
		 * @memberof module:mcfud/algo/NNetGA
		 * @param {function} optimizationFunction
		 * @param {function} isImprovement
		 * @param {function} isOptimal
		 * @param {function} getNextFeatureValue
		 * @param {any} initialFeatureValue
		 * @param {object} extra
		 * @return {object}
		 */
		function hillClimb(optimizationFunction, isImprovement,
											 isOptimal, getNextFeatureValue, initialFeatureValue,extra){
			let start= extra.startTime=_.now();
			let child,best = optimizationFunction(initialFeatureValue);
			while(!isOptimal(best)){
				child = optimizationFunction( getNextFeatureValue(best));
				if(isImprovement(best, child)){
					best = child
				}
			}
			extra.endTime=_.now();
			return best;
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		function tournament(create, crossOver, compete, sortKey, numParents=10, maxGenerations=100){
			let best,bestScore,parents,pool=[];
			for(let i=0,z=1+numParents*numParents;i<z;++i){
				pool.push([create(),[0,0,0]])
			}
			bestScore = pool[0];
			function getSortKey(x){
				return sortKey(x[0],
											 x[1][CompetitionResult.Win],
											 x[1][CompetitionResult.Tie],
											 x[1][CompetitionResult.Loss]);
			}
			function getSortKeys(a,b){
				let x= getSortKey(a),
						y= getSortKey(b);
				return x<y?-1:(x>y?1:0);
			}
			let generation = 0;
			while(generation < maxGenerations){
				generation += 1;
				for(let i=0;i<pool.length;++i){
					for(let j=0;j<pool.length;++j){
						if(i == j) continue;
						let [playera, scorea] = pool[i];
						let [playerb, scoreb] = pool[j];
						let result = compete(playera, playerb);
						scorea[result] += 1;
						scoreb[2 - result] += 1;
					}
				}
				pool.sort(getSortKeys).reverse();
				if(getSortKey(pool[0]) > getSortKey([best, bestScore])){
					[best, bestScore] = pool[0];
				}
				parents=[];
				for(let i=0;i<numParents.length;++i){
					parents.push(pool[i][0]);
				}
				pool=[];
				for(let i=0;i<parents.length;++i)
					for(let j=0;j<parents.length;++j){
						if(i !== j)
							pool.push([crossOver(parents[i], parents[j]), [0, 0, 0]]);
					}
				parents.forEach(p=> pool.push([p,[0,0,0]]));
				pool.push([create(), [0, 0, 0]]);
			}
			return best;
		}

		//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		const _$={

			NeuronLayer,
			Neuron,
			NeuralNet,

			runGASearch,
			runGACycle,

			calcStats,

			NumFitness,
			Fitness,
			Chromosome,

			mutateSM,
			mutateDM,
			mutateIVM,
			mutateDIVM,

			crossOverOBX,
			crossOverPBX,
			crossOverRND,
			crossOverPMX,
			crossOverAtSplits,

			hillClimb,

			//getChromoRoulette,
			//chromoRoulette,
			//tournamentSelectionN,
			//tournamentSelection,

			//fitnessScaleRank,
			//fitnessScaleSigma,
			//fitnessScaleBoltzmann,

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
				console.log("total generations= " + extra.gen);
        console.log("total cycles= " + extra.cycles);
        console.log("fitness= "+ best.fitness.score());
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
    module.exports=_module(require("../main/core"))
  }else{
    gscope["io/czlab/mcfud/algo/NNetGA"]=_module
  }

})(this)


