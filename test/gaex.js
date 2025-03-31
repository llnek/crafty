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
  function _module(Core,GA){

    if(!GA) GA= gscope["io/czlab/mcfud/algo/NNetGA"]();
    if(!Core) Core= gscope["io/czlab/mcfud/core"]();

    const int=Math.floor;
    const {is,u:_}= Core;

    const {NumFitness,Fitness, runGACycle,
           hillClimb, runGASearch,Chromosome,ChromoNumero, showBest,calcStats}= GA;

    /**
     * @module mcfud/algo/gaex
     */

    function _showBest(){

      console.log("best=" + best.applyGenes((gs)=> gs.join("")));
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Guess a password.
     * @memberof module:mcfud/algo/gaex
     * @class
     */
    class CH1{
      static test(input){
        let geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,".split("");
        let gs=geneSet.slice();
        let target=input.split(""),
            N=target.length, optimal= N;
        let params=GA.config({});

        function calcFit(guess){
          let sum=0;
          for(let i=0;i<N;++i)
            if(target[i]==guess[i]) ++sum;
          return sum;
        }

        function create(arg){
          return arg ? new ChromoNumero(arg, calcFit)
                     : new ChromoNumero( _.shuffle(gs).slice(0,N), calcFit)
        }

        function mutate(c){
          if(_.rand() < params.mutationRate){
            let i= _.randInt(c.length);
            c[i]= _.randItem(gs);
          }
        }

        function crossOver(b1,b2){
          return Chromosome.crossOverRND(b1,b2);
        }

        let tout, pop,best,s,extra;
        //console.log("ready...");
        if(0){
          extra= {gen:0,maxCycles:1000,create, mutate, crossOver, targetScore:12};
          [tout, pop]= runGACycle(100,extra);
          s=calcStats(pop);
          best=s.alpha;
        }

        if(1){
          extra= {gen:0,maxSeconds:35,maxCycles:1000, maxAge:50,create, calcFit, mutate, crossOver, poolSize:6};
          [tout,best]= runGASearch(optimal,extra);
        }

        showBest(best,extra,tout);
      }
    }
    //CH1.test("Hello World!");

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class CH3Chromo extends Chromosome{
      constructor(genes, calc, target){
        super(genes, calc, target);
        this.numSeq=0;
        this.gaps=0;
        this.recalcScore();
      }
      getScore(){ return this.numSeq }
      updateScore([f,g]){ this.numSeq=f; this.gaps=g; return this; }
      cmpScore([numSeq,gaps]){
        if(this.numSeq== numSeq){
          return this.gaps< gaps ? 1 : (this.gaps > gaps) ? -1 : 0;
        }else{
          return this.numSeq > numSeq ? 1 : -1;
        }
      }
      clone(){
        let [f,t]= this.getScoreCalcInfo();
        return new CH3Chromo(this._genes(), f, t);
      }
      compareTo(other){
        return this.cmpScore([other.numSeq,other.gaps]);
      }
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Find n numbers in sorted order.
     * @memberof module:mcfud/algo/gaex
     * @class
     */
    class CH3{
      static test(totalNumbers){
        let geneSet = _.fill(100,(i)=> i);
        let optimal= [totalNumbers, 0];
        let params= GA.config({});
        function calcFit(genes){
          let gap=0,fitness = 1;
          for(let i=1;i<genes.length;++i){
            if(genes[i] > genes[i-1])
              fitness += 1;
            else
              gap += genes[i-1] - genes[i];
          }
          return [fitness, gap];
        }
        function create(arg){
          return arg ? new CH3Chromo(arg,calcFit) :
                       new CH3Chromo( _.shuffle(geneSet,false).slice(0,totalNumbers), calcFit);
        }
        function mutate(c){
          if(_.rand() < params.mutationRate){
            let i= _.randInt(c.length);
            c[i]= _.randItem(geneSet);
          }
        }
        function crossOver(a,b){
          //return GA.crossOverPBX(a,b);//wont work!
          //return GA.crossOverRND(a,b);//ok
          //return GA.crossOverOBX(a,b);//ok
          return Chromosome.crossOverPMX(a,b);
        }
        let extra={ maxCycles:1000, create,mutate,crossOver,poolSize:6 };
        let [tout,best] = runGASearch(optimal,extra);
        showBest(best,extra,tout);
      }
    }
    //CH3.test(20);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class CH4Chromo extends Chromosome{
      #clashed;
      constructor(genes, calc, target){
        super(genes, calc, target);
        this.#clashed=0;
        this.recalcScore();
      }
      getScore(){ return this.#clashed }
      updateScore(c){ this.#clashed=c; return this; }
      cmpScore(s){
        return this.#clashed < s ? 1 : (this.#clashed > s  ? -1 : 0);
      }
      clone(){
        let [f,t]= this.getScoreCalcInfo();
        return new CH4Chromo(this._genes(), f, t);
      }
      compareTo(other){
        return this.cmpScore(other.getScore())
      }
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solve 8 queens.
     * @memberof module:mcfud/algo/gaex
     * @class
     */
    class CH4{
      static Board(genes, size){
        let board = _.fill(size, ()=> _.fill(size, "."));
        for(let row,col,i=0;i<genes.length;i+=2){
          row = genes[i];
          col= genes[i+ 1];
          board[row][col] = "Q";
        }
        return {
          get(row, col){ return board[row][col] },
          toString(){
            return board.reduce((acc,r)=>{
              return acc + r.join(" ") + "\n"
            },"")
          }
        }
      }
      static test(N){
        let params=GA.config({});
        function calcFit(genes){
          //best case is rows,cols dont overlap, so we would get
          //8 and 8 in those 2 maps. For diagonals, the two diags
          //are indexed by 2 formulas to check for overlaps
          let bd= CH4.Board(genes, N),
              NE = new Map(), SE= new Map(),
              ROWS = new Map(), COLS = new Map();
          for(let row=0;row<N;++row)
            for(let col=0;col<N;++col){
              if(bd.get(row, col) == "Q"){
                ROWS.set(row,1);
                COLS.set(col,1);
                NE.set(row+col,1);
                SE.set(N-1-row+col,1);
              }
            }
          let rc=N*4 - ROWS.size - COLS.size - NE.size - SE.size;
          return rc;
        }
        let geneSet = _.fill(N, (i)=> i);
        let optimal=0;
        function mutate(c){
          if(_.rand()<params.mutationRate){
            let i= _.randInt(c.length);
            c[i]= _.randItem(geneSet);
          }
        }
        function crossOver(a,b){
          return Chromosome.crossOverPBX(a,b);
        }
        let extra= {
          mutate,
          crossOver,
          create(arg){
            return arg ? new CH4Chromo(arg, calcFit) :
                         new CH4Chromo( _.shuffle(geneSet,false).concat(_.shuffle(geneSet,false)), calcFit);
          }
        }
        let [tout,best]=runGASearch(optimal, extra);
        showBest(best,extra,tout);
        console.log(CH4.Board(best._genes(), N).toString());
      }
    }
    //CH4.test(8);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solve Graph coloring.
     * @memberof module:mcfud/algo/gaex
     * @class
     */
    class CH5{
      static Edge(a,b){
        let left,right;
        if(a<b){ left=a; right=b; } else { left=b; right=a; }
        //console.log(`edge=${left}->${right}`);
        return{
          left,right,
          id(){ return `${left}:${right}` },
          isValid(genes, nodeIndexLookup){
            let i= nodeIndexLookup.get(left),
                j= nodeIndexLookup.get(right);
            return genes[i] != genes[j];
          }
        }
      }
      static test(){
        let params=GA.config({});
        let Rules=new Map(), Nodes=new Map();
        `AK AL,FL;GA;MS;TN AR,LA;MO;MS;OK;TN;TX AZ,CA;NM;NV;UT CA,AZ;NV;OR CO,KS;NE;NM;OK;UT;WY CT,MA;NY;RI DC,MD;VA
         DE,MD;NJ;PA FL,AL;GA GA,AL;FL;NC;SC;TN HI IA,IL;MN;MO;NE;SD;WI ID,MT;NV;OR;UT;WA;WY IL,IA;IN;KY;MO;WI
         IN,IL;KY;MI;OH KS,CO;MO;NE;OK KY,IL;IN;MO;OH;TN;VA;WV LA,AR;MS;TX MA,CT;NH;NY;RI;VT MD,DC;DE;PA;VA;WV
         ME,NH MI,IN;OH;WI MN,IA;ND;SD;WI MO,AR;IA;IL;KS;KY;NE;OK;TN MS,AL;AR;LA;TN MT,ID;ND;SD;WY NC,GA;SC;TN;VA
         ND,MN;MT;SD NE,CO;IA;KS;MO;SD;WY NH,MA;ME;VT NJ,DE;NY;PA NM,AZ;CO;OK;TX NV,AZ;CA;ID;OR;UT NY,CT;MA;NJ;PA;VT
         OH,IN;KY;MI;PA;WV OK,AR;CO;KS;MO;NM;TX OR,CA;ID;NV;WA PA,DE;MD;NJ;NY;OH;WV RI,CT;MA SC,GA;NC SD,IA;MN;MT;ND;NE;WY
         TN,AL;AR;GA;KY;MO;MS;NC;VA TX,AR;LA;NM;OK UT,AZ;CO;ID;NV;WY VA,DC;KY;MD;NC;TN;WV VT,MA;NH;NY WA,ID;OR
         WI,IA;IL;MI;MN WV,KY;MD;OH;PA;VA WY,CO;ID;MT;NE;SD;UT`.split(/\s+/).forEach(s=>{
           let p,e,a=s.split(/[,;]/).filter(x=>x.length>0); _.assert(a.length>0,"Boom");
           p=a.shift();
           Nodes.set(p,1);
           a.forEach(x=>{
             Nodes.set(x,1);
             e=CH5.Edge(p,x);
             Rules.set(e.id(),e);
           });
         });
        //console.log("Nodes===="+Nodes.size);
        let Colors=new Map(),
            NodeIndex=new Map(),
            optimal= Rules.size;
        let geneSet= ["Orange", "Yellow", "Green", "Blue"].map(c=>{
          Colors.set(c.charAt(0),c);
          return c.charAt(0);
        });
        let keys=Array.from(Nodes.keys()).sort();
        keys.forEach((n,i)=> NodeIndex.set(n,i));
        function calcFit(genes){
          return Rules.values().reduce((acc,r)=> acc + (r.isValid(genes, NodeIndex) ? 1 : 0) ,0)
        }
        function mutate(c){
          if(_.rand()<params.mutationRate){
            let i= _.randInt(c.length);
            c[i]= _.randItem(geneSet);
          }
        }
        function crossOver(a,b){
          return Chromosome.crossOverPBX(a,b);
        }
        function create(arg){
          return arg ? new ChromoNumero(arg, calcFit)
            : new ChromoNumero(_.fill(keys.length,0).map(x=> _.randItem(geneSet)), calcFit);
        }
        let extra={maxCycles:1000, mutate,crossOver,create};
        let [tout, best] = runGASearch(optimal, extra);
        showBest(best,extra,tout);
        keys.forEach((k,i)=>{
          console.log(`${k} is ${Colors.get(best.getGeneAt(i))}`);
        });
      }
    }
    //CH5.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class CH6Chromo extends Chromosome{
      constructor(genes, calc, target){
				super(genes, calc, target);

        this.totalDifference= 0;
        this.group1Sum=0;
        this.group2Product=0;
        this.duplicateCount=0;

				this.recalcScore();
			}
			getScore(){
        return [ this.group1Sum, this.group2Product, this.duplicateCount ];
      }
			updateScore([sum,prod,dups]){
        this.duplicateCount= dups;
        this.group2Product= prod;
        this.group1Sum= sum;
      }
      #xxx(sum,prod,dups){
        let productDifference= Math.abs(360 - prod);
        let sumDifference= Math.abs(36 - sum);
        return [sumDifference, productDifference, sumDifference + productDifference];
      }
			cmpScore([sum,prod,dups]){
        let [sdiff_m, pdiff_m, tdiff_m]= this.#xxx( this.group1Sum, this.group2Product, this.duplicateCount);
        let [sdiff_o, pdiff_o, tdiff_o]= this.#xxx( sum, prod, dups);
        if(this.duplicateCount != dups){
          return this.duplicateCount < dups ? 1 : -1;
        }else{
          return tdiff_m < tdiff_o ? 1 : (tdiff_m > tdiff_o ? -1 : 0) ;
        }
      }
			clone(){
				return new CH6Chromo(this._genes(), f, t);
			}
			compareTo(other){
        return this.cmpScore([other.group1Sum, other.group2Product, other.duplicateCount]);
			}
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solve a card problem.
     * @memberof module:mcfud/algo/gaex
     * @class
     */
    class CH6{
      static test(){
        let params=GA.config();
        //A,2-10
        let geneSet = _.fill(10,(i)=>i+1);
        function calcFit(genes){
          let g1Sum = 0,
              g2Prod = 1,
              duplicates = genes.length - new Set(genes).size;
          for(let i=0;i<5;++i) g1Sum += genes[i];
          for(let i=5;i<10;++i) g2Prod *= genes[i];
          return [g1Sum, g2Prod, duplicates];
        }
        function mutate(genes){
          if(_.rand()<params.mutationRate){
            if(genes.length == new Set(genes).size){
              let ix=_.fill(genes.length,(i)=> i);
              for(let i=0,count = _.randInt2(1, 4); i<count;++i){
                _.shuffle(ix);
                _.swap(genes,ix[0],ix[1]);
              }
            }else{
              genes[_.randInt(genes.length)] = geneSet[_.randInt(geneset.length)];
            }
          }
        }
        function create(arg){
          return arg ? new CH6Chromo(arg, calcFit)
            : new CH6Chromo( _.shuffle(geneSet,false), calcFit);
        }
        let extra={maxCycles:1000,mutate,create};
        let optimal= [36, 360, 0];
        let [tout,best]=runGASearch(optimal, extra);
        showBest(best,extra,tout);
        console.log(best._genes().slice(0,5).join(",") + " - " + best._genes().slice(5).join(","));
      }
    }
    //CH6.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solve Knight's problem.
     * @memberof module:mcfud/algo/gaex
     * @class
     */
    class CH7{
      static Position(x, y){
        return {x,y,
          toString(){ return `(${x}:${y})` },
          id(){ return `${x},${y}` }}
      }
      static Board(genes, N){
        //console.log(JSON.stringify(genes.slice(0,7)));
        //console.log(JSON.stringify(genes.slice(7)));
        let board=_.fill(N,()=> _.fill(N,"."));
        genes.forEach(p=> board[p.y][p.x]="N");
        return{
          toString(){
            return board.reduce((acc,r)=>{
              return acc + r.join(" ") + "\n"
            },"")
          }
        }
      }
      static test(N, expectedK){
        let params=GA.config();
        let optimal= N*N;
        function getAttacks(loc){
          let out=new Map(),
              X= [-2, -1, 1, 2],
              Y= [-2, -1, 1, 2];
          for(let p,x,i=0;i<X.length;++i){
            x=loc.x+X[i];
            if(x>=0 && x<N)
              for(let y,j=0;j<Y.length;++j){
                y=loc.y+Y[j];
                if(y>=0 && y<N && Math.abs(X[i]) != Math.abs(Y[j])){
                  p=CH7.Position(x,y);
                  out.set(p.id(),p);
                }
              }
          }
          return Array.from(out.values());
        }
        function calcFit(genes){
          let out=new Map();
          genes.forEach(k=> getAttacks(k).forEach(p=> out.set(p.id(),1)));
          return out.size;
        }
        let allPos=[];
        for(let x=0;x<N;++x)
          for(let y=0;y<N;++y) allPos.push(CH7.Position(x,y));
        let allPosMap=allPos.reduce((m,v)=>{
          m.set(v.id(),v); return m;
        },new Map());
        let nonEdgePos= allPos.map(p=>{
          return (p.x>0 && p.x < N-1 && p.y>0 && p.y < N-1)?p:null
        }).filter(o=> o != null);
        let nonEdgePosMap= nonEdgePos.reduce((m,v)=>{
          m.set(v.id(), 0); return m;
        },new Map());
        function mutate(genes){
          if(_.rand()>params.mutationRate){return}
          let cnt= _.randInt(10)==0 ? 2 : 1;
          let KI,unattacked,posToKIndexes,potentialKPos;
          while(cnt>0){
            cnt -= 1;
            unattacked = [];
            posToKIndexes= allPos.reduce((m,v)=>{
              m.set(v.id(),[]); return m;
            },new Map());
            genes.forEach((k,i)=>{
              getAttacks(k).forEach(p=> posToKIndexes.get(p.id()).push(i));
            });
            KI=genes.reduce((m,v,i)=>{ m.set(i,0); return m; },new Map());
            posToKIndexes.forEach((v,k)=>{
              if(v.length==0){
                unattacked.push(allPosMap.get(k));
              }else if(v.length==1){
                if(KI.has(v[0])) KI.delete(v[0])
              }
            });
            potentialKPos= nonEdgePos;
            if(unattacked.length>0){
              potentialKPos=[];
              for(let a,i=0; i<unattacked.length;++i){
                a=getAttacks(unattacked[i]);
                _.assert(a.length>0,"CRap");
                for(let p,j=0;j<a.length;++j){
                  p=a[j];
                  if(nonEdgePosMap.has(p.id())) potentialKPos.push(p);
                }
              }
            }
            let gi= KI.size ==0 ? _.randInt(genes.length)
                                : _.randItem(Array.from(KI.keys()));
            _.assert(potentialKPos.length>0,"Boom");
            genes[gi] = _.randItem(potentialKPos);
          }
        }
        function create(arg){
          return arg ? new ChromoNumero(arg,calcFit)
            : new ChromoNumero(_.fill(expectedK, ()=> _.randItem(nonEdgePos)), calcFit);
        }
        let extra={maxCycles:1000, mutate,create};
        let [tout,best] = runGASearch(optimal,extra);
        showBest(best,extra,tout);
        console.log(CH7.Board(best._genes(),N).toString());
      }
    }
    //CH7.test(8,14);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class CH8Chromo extends ChromoNumero{
			constructor(genes, calc, target){
				super(genes, calc, target);
			}
      cmpScore(s){ return this.getScore()<s ? 1 : (this.getScore()>s? -1 : 0) }
			clone(){
				let [f,t]= this.getScoreCalcInfo();
				return new CH8Chromo(this._genes(), f, t);
			}
		}
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solve Magic Square.
     * @memberof module:mcfud/algo/gaex
     * @class
     */
    class CH8{
      static getSums(genes, N){
        let se=0, ne=0,
            rows = _.fill(N,0),
            columns = _.fill(N,0);
        for(let v,r=0;r<N;++r){
          for(let c=0;c<N;++c){
            v= genes[r * N + c];
            rows[r] += v;
            columns[c] += v;
          }
          se += genes[r * N + r];
          ne += genes[r * N + (N - 1 - r)];
        }
        //console.log("rows=="+rows.join(","));
        //console.log("cols=="+columns.join(","));
        //console.log("ne==="+ne+", se==="+se);
        //console.log("=======");
        //console.log(_.partition(3,genes).join("\n"));
        return [rows, columns, ne, se];
      }
      static test(N,maxAge){
        let params=GA.config();
        let SQ= N*N;
        let optimal= 0;
        let geneSet = _.fill(SQ, (i)=> i+1);
        let expected= N* Math.floor((SQ+1)/2);
        let geneIndexes = _.fill(geneSet.length,(i)=> i);
        function calcFit(genes){
          let [rows, cols, ne, se]= CH8.getSums(genes, N),
              sums=rows.concat(cols).concat([se,ne]).filter(s=> s != expected);
          return sums.reduce((acc,v)=> acc + Math.abs(v-expected),0);
        }
        function mutate(genes){
          if(_.rand()<params.mutationRate){
            _.shuffle(geneIndexes);
            _.swap(genes, geneIndexes[0], geneIndexes[1]);
          }
        }
        function create(arg){
          return arg ? new CH8Chromo(arg, calcFit)
            : new CH8Chromo( _.shuffle(geneSet.slice()), calcFit);
        }
        let extra={ maxCycles:1000, maxAge, mutate, create};
        let [tout,best] = runGASearch(optimal, extra);
        showBest(best,extra,tout);
        for(let i=0;i<N;++i){
          let row=best._genes().slice(i*N, (i+1)*N);
          console.log(row.join(" "));
        }
      }
    }
    //CH8.test(3,500);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class CH9Chromo extends Chromosome{

			constructor(genes, calc, target){
				super(genes, calc, target);
        this.totalWeight=0;
        this.totalVolume=0;
        this.totalValue=0;
        this.recalcScore();
			}
      getScore(){
        return [ this.totalWeight, this.totalVolume, this.totalValue ];
      }
      updateScore([totalWeight, totalVolume, totalValue]){
        this.totalWeight= totalWeight;
        this.totalVolume= totalVolume;
        this.totalValue= totalValue;
      }
			cmpScore([totalWeight, totalVolume, totalValue]){
        if(this.totalValue != totalValue)
          return this.totalValue > totalValue ? 1 : -1;
        if(this.totalWeight != totalWeight)
          return this.totalWeight < totalWeight ? 1 : -1;
        return this.totalVolume < totalVolume ? 1 : (this.totalVolume > totalVolume ? -1 : 0);
      }
			clone(){
				let [f,t]= this.getScoreCalcInfo();
				return new CH9Chromo(this._genes(), f, t);
			}
      compareTo(o){
        return this.cmpScore(o.getScore());
      }
		}
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solve Knapsack problem.
     * @memberof module:mcfud/algo_go
     * @class
     */
    class CH9{
      static ItemQuantity(item, quantity){
        return{
          item,quantity,
          toString(){ return `{item=${this.item.name}, qty=${this.quantity}}` },
          eq(b){ return this.item === b.item && this.quantity == b.quantity }
        }
      }
      static Resource(name, value, weight, volume){
        return{name,value,weight,volume, eq(b){ return this===b; }}
      }
      static Window(min,max,size){
        return{
          min,max,size,
          slide(){ this.size = this.size>this.min? this.size - 1: this.max; }
        }
      }
      static test(){
        let params=GA.config();
        let geneSet=[CH9.Resource("Flour", 1680, 0.265, 0.41),
                     CH9.Resource("Butter", 1440, 0.5, 0.13),
                     CH9.Resource("Sugar", 1840, 0.441, 0.29) ];
        let MaxWeight = 10, MaxVolume = 4;
        let window = CH9.Window(1, int(Math.max(1, geneSet.length/3)), int(geneSet.length/2));
        let optimal = calcFit([CH9.ItemQuantity(geneSet[0], 1),
                               CH9.ItemQuantity(geneSet[1], 14),
                               CH9.ItemQuantity(geneSet[2], 6)]);
        geneSet.sort((a,b)=> a.value<b.value?-1:1);
        //console.log(optimal.toString())
        function calcFit(genes){
          let c,totalWeight = 0, totalVolume = 0, totalValue = 0;
          genes.forEach(g=>{
            c= g.quantity;
            totalWeight += g.item.weight * c;
            totalVolume += g.item.volume * c;
            totalValue += g.item.value * c;
          });
          return [totalWeight, totalVolume, totalValue];
        }
        function maxQ(item, maxWeight, maxVolume){
          return int(Math.min(maxWeight/item.weight, maxVolume/item.volume)) }
        function add(genes, items, maxWeight, maxVolume){
          let item;
          while(1){
            item= _.randItem(items);
            for(let g,j=0;j<genes.length;++j){
              g=genes[j];
              if(g.item===item){
                item=null;
                break;
              }
            }
            if(item) break;
          }
          let m= maxQ(item, maxWeight, maxVolume);
          return m>0? CH9.ItemQuantity(item, m) : null;
        }
        function create(arg){
          let genes = [],
              remainWeight=MaxWeight,
              remainVolume = MaxVolume,
              n=_.randInt2(1,geneSet.length);
          for(let g,i=0;i<n;++i){
            g = add(genes, geneSet, remainWeight, remainVolume);
            if(g){
              genes.push(g);
              remainWeight -= g.quantity * g.item.weight;
              remainVolume -= g.quantity * g.item.volume;
            }
          }
          return arg ? new CH9Chromo(arg, calcFit) : new CH9Chromo(genes,calcFit);
        }
        function mutate(genes){
          if(_.rand()>params.mutationRate){return}
          let [totalWeight, totalVolume, totalValue]= calcFit(genes),
              remainWeight = MaxWeight - totalWeight,
              remainVolume = MaxVolume - totalVolume,
              removing = genes.length>1 &&  _.randInt2(0, 10) == 0;
          let g,item,index;
          window.slide();
          if(removing){
            index = _.randInt(genes.length);
            g = genes[index];
            item = g.item;
            remainWeight += item.weight * g.quantity;
            remainVolume += item.volume * g.quantity;
            genes.splice(index,1);
          }
          if((remainWeight>0 || remainVolume>0) &&
             (genes.length == 0 || (genes.length < geneSet.length && _.randInt2(0, 100) == 0))){
            let n= add(genes, geneSet, remainWeight, remainVolume);
            if(n){
              genes.push(n);
              return;
            }
          }
          index = _.randInt(genes.length);
          g = genes[index];
          item = g.item;
          remainWeight += item.weight * g.quantity;
          remainVolume += item.volume * g.quantity;
          if(genes.length < geneSet.length && _.randInt2(0, 4) == 0){
            let itemIndex = geneSet.indexOf(g.item);
            let start = Math.max(1, itemIndex - window.size);
            let stop = Math.min(geneSet.length - 1, itemIndex + window.size);
            item = geneSet[_.randInt2(start, stop)];
          }
          let mQ= maxQ(item, remainWeight, remainVolume);
          if(mQ > 0){
            genes[index] = CH9.ItemQuantity(item, window.size>1?mQ: _.randInt2(1, mQ));
          }else{
            genes.splice(index,1);
          }
        }
        let extra={ maxCycles:1000, maxAge:50, mutate, create };
        let [tout,best] = runGASearch(optimal,extra);
        showBest(best,extra,tout);
        best._genes().forEach(g=>{
          console.log(`item=${g.item.name}, quantity=${g.quantity}`)
        });
      }
    }
    //CH9.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class CH10Chromo extends ChromoNumero{
			constructor(genes, calc, target){
				super(genes, calc, target);
			}
			cmpScore(s){ return this.getScore()<s ? 1 : (this.getScore()>s? -1 : 0) }
			clone(){
				let [f,t]= this.getScoreCalcInfo();
				return new CH10Chromo(this._genes(), f, t);
			}
		}
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solving Linear Equations.
     * @memberof module:mcfud/algo_go
     * @class
     */
    class CH10{
      static Window(min, max, size){
        return{
          min,max,size,
          slide(){
            this.size = this.size>this.min? this.size-1 : this.max;
          }
        }
      }
      static Fitness(totalDifference){
        return{
          totalDifference,
          toString(){ return `${this.totalDifference}` },
          eq(b){ return this.totalDifference==b.totalDifference},
          gt(b){ return this.totalDifference < b.totalDifference },
          lt(b){ return this.totalDifference > b.totalDifference },
          score(){ return this.totalDifference },
          clone(){
            return Fitness(this.totalDifference)
          }
        }
      }
      static test(numUnknowns){
        let params=GA.config();
        let geneSet = [-5,-4,-3,-2,-1,1,2,3,4];
        function fnGenesToInputs(genes){ return [genes[0], genes[1]] }
        function e1(genes){
          let [x, y] = fnGenesToInputs(genes);
          return x + 2 * y - 4;
        }
        function e2(genes){
          let [x, y] = fnGenesToInputs(genes);
          return 4 * x + 4 * y - 12;
        }
        let EQS=[ `x + 2 * y - 4 `, `4 * x + 4 * y - 12 ` ,""];
        let equations= [e1,e2];
        let MaxAge=50;
        let window = CH10.Window(Math.max(1, int(geneSet.length /(2 * MaxAge))),
                                 Math.max(1, int(geneSet.length / 3)), int(geneSet.length / 2));
        let geneIndexes = _.fill(numUnknowns,(i)=>i);
        let optimal= 0;
        function calcFit(genes){
          let v= equations.reduce((acc,e)=>{
            acc += Math.abs(e(genes));
            return acc;
          },0);
          return v;
        }
        function mutate(genes){
          if(_.rand()>params.mutationRate){return}
          let indexes;
          if(_.randInt2(0,10)==0)
            indexes=_.randSample(geneIndexes, _.randInt2(1,genes.length));
          else
            indexes=[_.randItem(geneIndexes)];
          window.slide();
          while(indexes.length>0){
            let index = indexes.pop();
            let genesetIndex = geneSet.indexOf(genes[index]);
            let start = Math.max(0, genesetIndex - window.size);
            let stop = Math.min(geneSet.length - 1, genesetIndex + window.size);
            genesetIndex = _.randInt2(start, stop);
            genes[index] = geneSet[genesetIndex];
          }
        }
        let extra={ maxAge:MaxAge, maxCycles:1000, mutate, create};
        function create(arg){
          return arg ? new CH10Chromo(arg, calcFit)
            : new CH10Chromo(_.shuffle(geneSet.slice()).slice(0,numUnknowns), calcFit);
        }
        let [tout,best] = runGASearch(optimal, extra);
        showBest(best,extra,tout);
        console.log(best._genes().join(","));
        console.log(EQS.join("=0\n"));
      }
    }
    //CH10.test(2);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solving Suduko.
     * @memberof module:mcfud/algo_go
     * @class
     */
    class CH11{
      static Rule(it, other){
        let r={
          eq(b){ return this.index == b.index && this.otherIndex == b.otherIndex; }
        };
        if(it > other){
          r.index = other; r.otherIndex = it;
        }else{
          r.index = it; r.otherIndex = other;
        }
        return r;
      }
      static test(){
        let params=GA.config();
        let geneSet = _.fill(9,(i)=>i+1);
        let optimal= 100;
        function indexRow(index){ return int(index / 9)}
        function indexColumn(index){ return index % 9 }
        function rowColumnSection(row, column){ return int(row / 3) * 3 + int(column / 3) }
        function indexSection(index){ return rowColumnSection(indexRow(index), indexColumn(index)) }
        function sectionStart(index){ return int((indexRow(index) % 9) / 3) * 27 + int(indexColumn(index) / 3) * 3 }
        function buildValidationRules(){
          let i,j,rules = [];
          for(i=0;i<80;++i){
            let itsRow = indexRow(i);
            let itsColumn = indexColumn(i);
            let itsSection = rowColumnSection(itsRow, itsColumn);
            for(j=i+1;j<81;++j){
              let otherRow = indexRow(j);
              let otherColumn = indexColumn(j);
              let otherSection = rowColumnSection(otherRow, otherColumn);
              if(itsRow == otherRow ||
                 itsColumn == otherColumn || itsSection == otherSection)
                rules.push(CH11.Rule(i, j));
            }
          }
          rules.sort((a,b)=>{
            let x=a.otherIndex*100+a.index;
            let y=b.otherIndex*100+b.index;
            return x<y?-1:(x>y?1:0);
          });
          //console.log(JSON.stringify(rules));
          return rules;
        }
        let validationRules = buildValidationRules();
        function fRule(genes){
          for(let r,i=0;i<validationRules.length;++i){
            r=validationRules[i];
            if(genes[r.index]==genes[r.otherIndex])
              return r;
          }
        }
        function calcFit(genes){
          let f=100, R= fRule(genes);
          if(R){
            f= (1 + indexRow(R.otherIndex)) * 10  + (1 + indexColumn(R.otherIndex));
          }
          return f;
        }
        function shuffleInPlace(genes, first, last){
          while(first < last){
            let index = _.randInt2(first, last);
            _.swap(genes, first,index);
            first += 1;
          }
        }
        function mutate(genes){
          if(_.rand()>params.mutationRate){return}
          let selectedRule = fRule(genes);
          if(!selectedRule)
            return;
          if(indexRow(selectedRule.otherIndex) % 3 == 2 && _.randInt2(0, 10) == 0){
            let sectStart = sectionStart(selectedRule.index);
            let current = selectedRule.otherIndex;
            while(selectedRule.otherIndex == current){
              shuffleInPlace(genes, sectStart, 80);
              selectedRule = fRule(genes);
            }
            return ;
          }
          let row = indexRow(selectedRule.otherIndex);
          let start = row * 9;
          _.swap(genes, selectedRule.otherIndex, _.randInt2(start, genes.length-1));
        }
        let extra={ maxAge:50, maxCycles:1000, mutate, create};
        function create(arg){
          return arg ? new ChromoNumero(arg, calcFit)
            : new ChromoNumero( _.shuffle(_.fill(9,0).map(x=> _.fill(9,(i)=>i+1)).flat()), calcFit);
        }
        let [tout,best] = runGASearch(optimal,extra);
        showBest(best,extra,tout);
        _.partition(9, best._genes()).forEach(r=> console.log(r.join(",")));
      }
    }
    CH11.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class CH12Chromo extends ChromoNumero{
			constructor(genes, calc, target){
				super(genes, calc, target);
			}
			cmpScore(s){ return this.getScore()<s ? 1 : (this.getScore()>s? -1 : 0) }
			clone(){
				let [f,t]= this.getScoreCalcInfo();
				return new CH12Chromo(this._genes(), f, t);
			}
		}
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solving Travel Salesman's Problem.
     * @memberof module:mcfud/algo_go
     * @class
     */
    class CH12{
      static test(){
        let params=GA.config();
        function getDist(locationA, locationB){
          let sideA = locationA[0] - locationB[0];
          let sideB = locationA[1] - locationB[1];
          return Math.sqrt(sideA * sideA + sideB * sideB);
        }
        function calcFit(genes){
          let d = getDist(idToLocationLookup[genes[0]],
                          idToLocationLookup[_.last(genes)]);
          for(let s,e,i=0;i<genes.length-1;++i){
            s= idToLocationLookup[genes[i]];
            e= idToLocationLookup[genes[i + 1]];
            d += getDist(s, e);
          }
          return Math.round(d);
        }
        function mutate(genes){
          if(_.rand()>params.mutationRate){return}
          let count = _.randInt2(2, genes.length);
          let f0 = calcFit(genes);
          let ix= _.fill(genes.length,(i)=>i);
          while(count > 0){
            count -= 1;
            _.shuffle(ix);
            _.swap(genes, ix[0],ix[1]);
            if(calcFit(genes) < f0) return;
          }
        }
        function Pair(a,b){
          if(a<b){ let t=a; a=b;b=a; }
          return{a,b,id(){return `${this.a},${this.b}`}}
        }
        function crossOver(p1, p2){
          let parentGenes=p1.copyGenes();
          let donorGenes=p2.copyGenes();
          let p,pairs=new Map();
          p=Pair(donorGenes[0],_.last(donorGenes));
          pairs.set(p.id(),[0,p]);
          for(let i=0;i<donorGenes.length-1;++i){
            p=Pair(donorGenes[i], donorGenes[i+1]);
            pairs.set(p.id(),[0,p]);
          }
          let hit,tempGenes = parentGenes.slice();
          p=Pair(parentGenes[0], _.last(parentGenes));
          if(pairs.has(p.id())){
            //find a discontinuity
            hit = false;
            for(let i=0;i<parentGenes.length-1;++i){
              p=Pair(parentGenes[i], parentGenes[i+1]);
              if(pairs.has(p.id())) continue;
              tempGenes = parentGenes.slice(i+1).concat(parentGenes.slice(0,i+1));
              hit = true;
              break;
            }
            if(!hit) return [parentGenes, donorGenes];
          }
          let runs = [[tempGenes[0]]];
          for(let i=0;i<tempGenes.length-1;++i){
            p=Pair(tempGenes[i], tempGenes[i+1]);
            if(pairs.has(p.id())){
              _.last(runs).push(tempGenes[i + 1]);
              continue;
            }
            runs.push([tempGenes[i + 1]]);
          }
          let f0= calcFit(parentGenes);
          let count = _.randInt2(2, 20);
          let runIndexes= _.fill(runs.length,(i)=>i);
          let cg, rix=runIndexes.slice();
          while(count > 0){
            count -= 1;
            for(let i,x=0;x<runIndexes.length;++x){
              i=runIndexes[x];
              if(runs[i].length == 1) continue;
              if(_.randInt2(0, runs.length) == 0) runs[i] = runs[i].reverse();
            }
            if(runs.length>1){
              _.shuffle(rix);
              _.swap(runs, rix[0], rix[1]);
            }
            cg=[];
            runs.forEach(r=> cg.push(...r));
            if(calcFit(cg) < f0) return [cg,cg];
          }
          cg.forEach((v,i)=>{
            parentGenes[i]=v;
            donorGenes[i]=v;
          });

          return [parentGenes,donorGenes];
        }
        let geneSet = "ABCDEFGH".split("");
        let idToLocationLookup = {
          "A": [4, 7], "B": [2, 6], "C": [0, 5], "D": [1, 3],
          "E": [3, 0], "F": [5, 1], "G": [7, 2], "H": [6, 4] };
        let optimal= calcFit(geneSet);
        let extra={ maxCycles:1000, maxAge:500, poolSize:25, crossOver, mutate, create};
        function create(arg){
          return arg ? new CH12Chromo(arg, calcFit)
            : new CH12Chromo( _.shuffle(geneSet,false), calcFit);
        }
        let [tout,best] = runGASearch(optimal, extra);
        //console.log("optimal= " + optimal);
        showBest(best,extra,tout);
        console.log(best._genes().join(" -> "));
      }
    }
    //CH12.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Solving Approximation of PI.
     * @memberof module:mcfud/algo_go
     * @class
     */
    class CH13{
      static test(){
        let params=GA.config();
        let bitValues=[512, 256, 128, 64, 32, 16, 8, 4, 2, 1];
        let numBits= bitValues.length;
        let geneSet = [0,1];
        let optimal=  3.14159;
        function calcFit(genes){
          let denominator = getDenominator(genes, bitValues);
          if(denominator == 0) return 0;
          let ratio = getNumerator(genes, bitValues) / denominator;
          return Math.PI - Math.abs(Math.PI - ratio);
        }
        function bitsToInt(bits){
          let result = 0;
          for(let bit,i=0;i<bits.length;++i){
            bit=bits[i];
            if(bit == 0) continue;
            result += bitValues[i];
          }
          return result;
        }
        function getNumerator(genes){
          return 1 + bitsToInt(genes.slice(0,bitValues.length));
        }
        function getDenominator(genes){
          return bitsToInt(genes.slice(bitValues.length));
        }
        function mutate(genes){
          if(_.rand()<params.mutationRate){
            let numeratorIndex = _.randInt2(0, numBits-1);
            let denominatorIndex= _.randInt2(numBits, genes.length-1);
            genes[numeratorIndex] = 1 - genes[numeratorIndex];
            genes[denominatorIndex] = 1 - genes[denominatorIndex];
          }
        }
        let extra={ maxCycles:1000, maxAge:250, create,mutate};
        function create(arg){
          return arg ? new ChromoNumero(arg, calcFit)
            : new ChromoNumero( _.fill(bitValues.length*2,0).map(x=> _.randItem(geneSet)), calcFit);
        }
        let [tout,best] = runGASearch(optimal, extra);
        showBest(best,extra,tout);
      }
    }
    //CH13.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Not{
      constructor(input){
        this.input = input
      }
      getOutput(){
        let v;
        if(_.echt(this.input))
          v=this.input.getOutput();
        if(_.echt(v)) return !v;
      }
      toString(){
        return this.input?`Not(${this.input})` : "Not(?)"
      }
      static inputCount(){ return 1 }
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class GateWith2Inputs{
      constructor(inputA, inputB, label, fnTest){
        this.inputA = inputA;
        this.inputB = inputB;
        this.label = label;
        this.fnTest = fnTest;
      }
      getOutput(){
        let a,b;
        if(_.echt(this.inputA) && _.echt(this.inputB)){
          a= this.inputA.getOutput();
          b= this.inputB.getOutput();
        }
        if(_.echt(a) && _.echt(b)) return this.fnTest(a, b);
      }
      static inputCount(){ return 2 }
      toString(){
        return (_.nichts(this.inputA) ||
                _.nichts(this.inputB)) ? `${this.label}` : `${this.label}(${this.inputA} ${this.inputB})`
      }
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class And extends GateWith2Inputs{
      constructor(inputA, inputB){
        super(inputA, inputB, "And", (a,b)=> a && b)
      }
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Or extends GateWith2Inputs{
      constructor(inputA, inputB){
        super(inputA, inputB, "Or", (a,b)=> a || b)
      }
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Xor extends GateWith2Inputs{
      constructor(inputA, inputB){
        super(inputA, inputB, "Xor", (a,b)=> a != b)
      }
    }
    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Source{
      constructor(sourceId, sourceContainer){
        this.sourceId = sourceId;
        this.sourceContainer = sourceContainer;
      }
      getOutput(){
        return this.sourceContainer.get(this.sourceId)
      }
      toString(){
        return this.sourceId
      }
      static inputCount(){ return 0 }
    }

    class Circuits{
      static Node(createGate, indexA, indexB){ return{
        toString(){
          return `A=${this.indexA===undefined?"?":this.indexA},B=${this.indexB===undefined?"?":this.indexB}`
        },
        createGate, indexA, indexB }
      }
      static test(){
        let params=GA.config();
        let inputs = new Map();
        let maxLength=50;
        let rules = [[[false, false], false],
                 [[false, true], true],
                 [[true, false], true],
                 [[true, true], true]];
        let expectedLength = 6;
        let gates = [[(i1,i2)=> new And(i1,i2), And],
                     [(i1,i2)=> new Not(i1), Not]];
        let sources = [[(i1,i2)=> new Source("A", inputs), Source],
                       [(i1,i2)=> new Source("B", inputs), Source]];
        function nodesToCircuit(genes){
          let circuit = [], usedIndexes = [];
          for(let node,i=0;i<genes.length;++i){
            let inputA, inputB;
            node=genes[i];
            let used= new Set([i]);
            if(_.echt(node.indexA) && i > node.indexA){
              inputA = circuit[node.indexA];
              used.add(usedIndexes[node.indexA]);
              if(_.echt(node.indexB) && i > node.indexB){
                inputB = circuit[node.indexB];
                used.add(usedIndexes[node.indexB]);
              }
            }
            circuit.push(node.createGate(inputA, inputB));
            usedIndexes.push(Array.from(used));
          }
          return[_.last(circuit), _.last(usedIndexes)];
        }
        function calcFit(genes){
          let circuit = nodesToCircuit(genes)[0];
          let sourceLabels = "ABCD";
          let rulesPassed = 0;
          rules.forEach(rule=>{
            inputs.clear();
            _.zip(sourceLabels, rule[0], inputs);
            if(circuit.getOutput() == rule[1]) ++rulesPassed;
          });
          return rulesPassed;
        }
        function createGene(index){
          let gateType= index<sources.length ? sources[index] : _.randItem(gates);
          let indexA, indexB;
          if(gateType[1].inputCount() > 0)
            indexA = _.randInt2(0, index);
          if(gateType[1].inputCount() > 1){
            indexB = _.randInt2(0, index);
            if(indexB == indexA)
              indexB = _.randInt2(0, index);
          }
          //console.log("pppp="+index);
          return Circuits.Node(gateType[0], indexA, indexB);
        }
        function mutate(childGenes, fnCreateGene, fnGetFitness, sourceCount){
          let count = _.randInt2(1, 5);
          let f0 = fnGetFitness(childGenes);
          while(count > 0){
            count -= 1;
            let indexesUsed = nodesToCircuit(childGenes)[1].reduce((acc,i)=>{
              if(i>=sourceCount) acc.push(i);
              return acc;
            },[]);
            if(indexesUsed.length == 0) return;
            let index = _.randItem(indexesUsed);
            childGenes[index] = fnCreateGene(index);
            if(fnGetFitness(childGenes) > f0) return;
          }
        }
        let optimal= rules.length;
        //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        function fnOptimizationFunction(variableLength, extra){
          maxLength = variableLength;
          return runGASearch(optimal, extra)[1];
        }
        function fnCreate(arg){
          return arg ? new ChromoNumero(arg, calcFit)
            : new ChromoNumero( _.fill(maxLength,0).map((v,i)=> createGene(i)), calcFit);
        }
        function fnMutate(genes){
          if(_.rand()<params.mutationRate)
            mutate(genes, createGene, calcFit, sources.length)
        }
        //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        function fnIsImprovement(currentBest, child){
          return child.getScore()==rules.length &&
            nodesToCircuit(child._genes())[1].length < nodesToCircuit(currentBest._genes())[1].length
        }
        //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        function fnIsOptimal(child){
          return child.getScore() == rules.length &&
            nodesToCircuit(child._genes())[1].length <= expectedLength;
        }
        //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        function fnGetNextFeatureValue(currentBest){
          return nodesToCircuit(currentBest._genes())[1].length
        }

        //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        let extra={ maxCycles:100, mutate: fnMutate, create: fnCreate, poolSize:3 }
        let [tout, best] = GA.hillClimb(fnOptimizationFunction, fnIsImprovement,
                                        fnIsOptimal, fnGetNextFeatureValue, maxLength, extra);
        let cc=best ? nodesToCircuit(best._genes())[0] : "Op failed!";
        GA.showBest(best,extra);
        console.log(`${cc}`);
      }
    }
    //Circuits.test();

    const _$={ };
    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud")["Core"],require("../src/algo/genetic"))
  }else{
    gscope["io/czlab/mcfud/algo/gaex"]=_module
  }

})(this);


