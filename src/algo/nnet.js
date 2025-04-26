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

    const Params={
      BIAS: 1,
      actFunc:"sigmoid"
    };

    /**
     * @module mcfud/algo/NNet
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const NodeType={
      INPUT: 1, BIAS: 2, OUTPUT: 3, HIDDEN: 4, NONE: 911,
      toStr(t){
        switch(t){
          case NodeType.OUTPUT: return "output";
          case NodeType.INPUT: return "input";
          case NodeType.BIAS: return "bias";
          case NodeType.HIDDEN: return "hidden";
        }
        return "none";
      }
    };
    const FuncType=[
      "sigmoid",
      "tanh",
      "linear",
      "relu",
      "leaky_relu",
      "step",
      "swish",
      "softmax",
      "softplus"];
    const FuncTypeDB={
      sigmoid:function(x){
        return 1 / (1 + Math.exp(-x));
      },
      tanh:function(x){
        //let a=Math.exp(x), b= Math.exp(-x); return (a-b)/(a+b);
        //return 2 * FuncType.SIGMOID(2 * x) - 1;
        return Math.tanh(x);
      },
      linear:function(x){
        return x;
      },
      relu:function(x){
        return Math.max(0,x);
      },
      leaky_relu:function(x, alpha=0.01){
        return x>0 ? x : alpha * x;
      },
      step:function(x){
        return x>=0 ? 1 : 0;
      },
      swish:function(x){
        return x * FuncType.SIGMOID(x);
      },
      softmax:function(logits){
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
      softplus:function(x){
        return Math.log(1+ Math.exp(x));
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    function _isOUTPUT(n){ return n.type == NodeType.OUTPUT }
    function _isINPUT(n){ return n.type == NodeType.INPUT }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Coord{
      #x;
      #y;
      get x(){ return this.#x }
      get y(){ return this.#y }
      constructor(x=0,y=0){ this.#x=x; this.#y=y; }
      clone(){ return new Coord(this.#x, this.#y) }
      toJSON(){ return {x: this.#x, y: this.#y } }
      static dft(){ return new Coord(0,0) }
      static fromJSON(json){ return new Coord(json.x, json.y) }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Link{
      #fromNode;
      #toNode;
      #weight;
      /**
      */
      get fromNode(){ return this.#fromNode }
      get toNode(){ return this.#toNode }
      get weight(){ return this.#weight }
      set weight(w){ this.#weight=w }
      /**
       * @param {number} w
       * @param {Node} from
       * @param {Node} to
       */
      constructor(w, from, to){
        this.#fromNode=from;
        this.#toNode=to;
        this.#weight=w;
      }
      clone(){
        return new Link(this.weight, this.fromNode, this.toNode);
      }
      toJSON(){
        return {
          fromNode: this.#fromNode.id,
          toNode: this.#toNode.id,
          weight: this.#weight
        }
      }
      static add(from,to){ return new Link(_.randMinus1To1(), from, to) }
      static fromJSON(json, resolver){
        return new Link(json.weight, resolver(json.fromNode), resolver(json.toNode));
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Node{
      #vecLinksOut;
      #vecLinksIn;
      #inputSum;
      #actFunc;
      #output;
      #layer;
      #error;
      #type;
      #bias;
      #pos;
      #id;
      /**
      */
      get outputValue() { return this.#output }
      get inputSum(){ return this.#inputSum }
      get type() { return this.#type }
      get id() { return this.#id }
      getBias() { return this.#bias }
      get actFunc(){ return this.#actFunc }
      get posY(){ return this.#pos.y }
      get posX(){ return this.#pos.x }
      get pos(){ return this.#pos }
      get layer(){ return this.#layer }
      get errorValue(){ return this.#error }
      set outputValue(o) { this.#output=o }
      setBias(b) { this.#bias=b; return this }
      /**
       * @param {number} id
       * @param {NeuronType} type
       * @param {number} layer
       * @param {Coord} pos
       */
      constructor(id,type, layer, pos=null){
        this.#pos= pos ? pos.clone() : Coord.dft();
        this.#bias= _.randMinus1To1();
        this.#actFunc= "sigmoid";
        this.#layer=layer;
        this.#type=type;
        this.#error=0;
        this.#id=id;
        this.#output=0;
        this.#inputSum=0;
        this.#vecLinksOut=[];
        this.#vecLinksIn=[];
      }
      resetErrorValue(v){
        this.#error=v; return this; }
      toJSON(){
        return {
          pos: {x: this.#pos.x, y: this.#pos.y},
          bias: this.#bias,
          type: this.#type,
          id: this.#id,
          layer: this.#layer,
          output: this.#output,
          inputSum: this.#inputSum,
          actFunc: this.#actFunc,
          vecLinksOut: this.#vecLinksOut.map(k=> k.toJSON())
        };
      }
      /**Resets everything.
       * @return
       */
      flush(){
        _.trunc(this.#vecLinksOut);
        this.#inputSum=0;
        this.#output=0;
        return this;
      }
      /**Change the value of total inputs.
       * @param {number} n
       * @return
       */
      resetInput(n=0){
        this.#inputSum=n; return this; }
      /**Add value to the total inputs.
       * @param {number} n
       * @return
       */
      addInput(n){
        this.#inputSum += n; return this; }
      /**Add a output connection - linking to another node.
       * @param {Link} k
       * @return
       */
      addOutLink(k){
        this.#vecLinksOut.push(k); return this; }
      addInLink(k){
        this.#vecLinksIn.push(k); return this; }
      /**Push value downstream to all the output connections.
       * @param {function|string} fn activation function
       * @return {Node} this
       */
      activate(fn){
        if(this.#type != NodeType.INPUT){
          fn= (fn || this.#actFunc || Params.actFunc);
          if(is.str(fn)) fn= FuncTypeDB[fn];
          _.assert(is.fun(fn), "activation function not found");
          this.#output= fn(this.#inputSum + this.#bias);
        }
        this.#vecLinksOut.forEach(k=> k.toNode.addInput(k.weight * this.#output));
        return this;
      }
      iterInLinks(f,target){
        this.#vecLinksIn.forEach(f, target);
        return this;
      }
      iterOutLinks(f,target){
        this.#vecLinksOut.forEach(f, target);
        return this;
      }
      findLinkIn(from){
        return this.#vecLinksIn.find(k=> k.fromNode.id== from.id);
      }
      setActFunc(aFunc){
        this.#actFunc=aFunc; return this; }
      _cpy(af, bias, inLinks, outLinks){
        this.#vecLinksOut=outLinks.map(v=> v.clone());
        this.#vecLinksIn=inLinks.map(v=> v.clone());
        this.#bias=bias;
        this.#actFunc=af;
        return this;
      }
      clone(){
        return new Node(this.id,this.type,this.pos).
               _cpy(this.#actFunc, this.#bias, this.#vecLinksIn,this.#vecLinksOut)
      }
      static fromJSON(json){
        let nn=new Node(json.id, json.type, json.layer, Coord.fromJSON(json.pos));
        nn.setActFunc(json.actFunc);
        nn.outputValue= json.output ?? 0;
        nn.resetInput(json.inputSum || 0);
        nn.bias= json.bias ?? _.randMinus1To1();
        return nn;
      }
    }

    class Trainer{
      #tolerance;
      #learnRate;
      #errorSum;
      #cycle;
      #status;
      get cycle(){return this.#cycle}
      get status(){return this.#status}
      get errorSum(){ return this.#errorSum }
      get learnRate(){return this.#learnRate}
      get tolerance(){return this.#tolerance}
      /**
      */
      constructor(learnRate,tolerance){
        this.#tolerance= tolerance ?? 0;
        this.#learnRate= learnRate ?? 0;
        this.#status= false;
        this.#errorSum=0;
        this.#cycle=0;
      }
      setStatus(s){
        this.#status=s; return this; }
      setErrorSum(s){
        this.#errorSum = s; return this }
      addErrorSum(e){
        this.#errorSum += e; return this }
      addCycle(){
        this.#cycle +=1; return this }
      resetCycle(){
        this.#cycle=0; return this }
      /**
      */
      toJSON(){
        return {
          tolerance: this.#tolerance,
          errorSum: this.#errorSum,
          status: this.#status,
          cycle:this.#cycle,
          learnRate: this.#learnRate
        }
      }
      /**
      */
      static fromJSON(j){
        return new Trainer(j.learnRate, j.tolerance).
          setErrorSum(j.errorSum).
          setStatus(j.status).
          setCycle(j.cycle);
      }
    }


    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class NeuralNet{
      #vecNodes;
      #trainer;
      #inputs;
      #outputs;
      get errorSum(){ return this.#trainer.errorSum }
      get trainCycle(){return this.#trainer.cycle}
      get trained(){return this.#trainer.status}
      get trainer(){return this.#trainer}
      get numOutputs(){return this.#outputs}
      get numInputs(){return this.#inputs}
      /**
       * @param {number} inputs
       * @param {number} outputs
       * @param {object} options
       */
      constructor(inputs,outputs,options){

        let nObj, NID=0, layers=2,
            iXGap= 1/(inputs+2), oXGap= 1/(outputs+1);

        options= options || {};

        this.#trainer=new Trainer();
        this.#outputs=outputs;
        this.#inputs=inputs;
        this.#vecNodes=[];

        if(options==="json"){
          return;
        }

        if(options.layers)
          layers += options.layers.length;

        for(let i=0; i<inputs; ++i){
          nObj=new Node(++NID, NodeType.INPUT, 0, new Coord((i+2)*iXGap, 0));
          this.#vecNodes.push(nObj);
        }

        for(let i=0; i<outputs; ++i){
          nObj=new Node(++NID, NodeType.OUTPUT, layers-1, new Coord((i+1)*oXGap, 1));
          this.#vecNodes.push(nObj.setActFunc(options.actOut));
        }

        if(options.layers){
          let posY=0, gaps= 1/(options.layers.length+1);
          options.layers.forEach((o,py)=>{
            iXGap= 1/(o.size+2);
            posY += gaps;
            for(let i=0; i< o.size; ++i){
              nObj=new Node(++NID, NodeType.HIDDEN, py+1, new Coord((i+2)*iXGap, posY));
              this.#vecNodes.push(nObj.setActFunc(o.actFunc));
            }
          });
        }

        //sort the nodes in the right order
        this.#vecNodes.sort(_.comparator(_.SORT_ASC, a=>a.posY, b=>b.posY));

        //link up the whole thing
        for(let a,b,i=0; i<= layers-2; ++i){
          a=this.#vecNodes.filter(n=> n.layer== i);
          b=this.#vecNodes.filter(n=> n.layer== (i+1));
          a.forEach(x=> b.forEach(o=>{
            let k=Link.add(x,o);
            x.addOutLink(k);
            o.addInLink(k);
          }));
        }

        if(0){
          console.log("Debug NeuralNet...");
          this.#vecNodes.forEach(n=> console.log(n.toJSON()));
        }
      }
      countLayers(){
        return this.#vecNodes.find(n=> _isOUTPUT(n)).layer + 1;
      }
      iterOutputLayer(f, target){
        this.#vecNodes.filter(n=> _isOUTPUT(n)).forEach(f, target);
        return this;
      }
      iterInputLayer(f,target){
        this.#vecNodes.filter(n=> _isINPUT(n)).forEach(f, target);
        return this;
      }
      iterNodes(f,target){
        this.#vecNodes.forEach(f, target);
        return this;
      }
      iterLayer(n, f, target){
        this.#vecNodes.filter(o=> o.layer==n).forEach(f, target);
        return this;
      }
      resetTraining(learnRate, errorSum, tolerance){
        this.#trainer=new Trainer(learnRate, tolerance);
        this.#trainer.setErrorSum(errorSum);
        this.#vecNodes.forEach(n=>{
          n.iterOutLinks(k=> k.weight= _.randMinus1To1())
        })
        return this;
      }
      trainedOneCycle(){
        this.#trainer.addCycle(); return this;
      }
      addError(e){
        this.#trainer.addErrorSum(e); return this; }
      resetErrorSum(n=0){
        this.#trainer.setErrorSum(n); return this; }
      checkTraining(){
        let rc= this.#trainer.errorSum > this.#trainer.tolerance ? false : true;
        if(rc)
          this.#trainer.setStatus(true);
        return rc;
      }
      clone(){
        return NeuralNet.fromJSON(this.toJSON())
      }
      size(){
        return this.#vecNodes.length;
      }
      /**Update network for this clock cycle.
       * @param {number[]} data
       * @param {RunType} type
       * @return {number[]} outputs
       */
      compute(data){ return this.update(data) }
      /**Update network for this clock cycle.
       * @param {number[]} data
       * @param {RunType} type
       * @return {number[]} outputs
       */
      update(data){
        _.assert(data.length==this.#inputs,
          `update: expecting ${this.#inputs} inputs but got ${data.length}`);
        this.#vecNodes.forEach((n,i)=> n.type==NodeType.INPUT ? n.outputValue= data[i] : 0);
        this.#vecNodes.forEach(n=> n.activate());
        let outs= this.#vecNodes.reduce((acc,n)=>{
          if(n.type==NodeType.OUTPUT) acc.push(n.outputValue);
          return acc;
        },[]);
        this.#vecNodes.forEach(n=> n.resetInput(0));
        return outs;
      }
      /**
      */
      _injectFromJSON(nodes){
        _.append(this.#vecNodes, nodes, true); return this; }
      /**
      */
      _injectTrainer(j){
        this.#trainer= new Trainer(j.learnRate, j.tolerance);
        return this;
      }
      /**
      */
      toJSON(){
        let o, arr=[], json={
          trainer: this.#trainer.toJSON(),
          outputs: this.#outputs,
          inputs: this.#inputs,
          nodes: this.#vecNodes.map(n=>{
            o=n.toJSON();
            o.vecLinksOut.forEach(k=> arr.push(k));
            delete o.vecLinksOut;
            return o;
          }),
          links: []
        };
        _.append(json.links,arr,true);
        return json;
      }
      /**
      */
      static fromJSON(json){
        let nnet= new NeuralNet(json.inputs, json.outputs, "json");
        let a,b,o, vs=[], m= new Map();
        function rs(id){ return m.get(id) }
        json.nodes.forEach(n=>{
          o= Node.fromJSON(n);
          m.set(o.id, o);
          vs.push(o);
        });
        nnet._injectFromJSON(vs.sort(_.comparator(_.SORT_ASC, a=>a.posY, b=>b.posY)));
        nnet._injectTrainer(json.trainer);
        json.links.forEach((k,i)=>{
          rs(k.fromNode).addOutLink(i=Link.fromJSON(k, rs));
          rs(k.toNode).addInLink(i);
        });
        return nnet;
      }
      /**
      */
      static trainOneCycle(nnet, setIn, setOut){
        let err, outputs, learnRate= nnet.trainer.learnRate;
        nnet.resetErrorSum();
        for(let vec=0;vec<setIn.length;++vec){

          outputs = nnet.update(setIn[vec]);
          if(outputs.length==0) return false;

          nnet.iterOutputLayer((u,op)=>{
            err = (setOut[vec][op] - outputs[op]) * outputs[op] * (1 - outputs[op]);
            u.resetErrorValue(err);
            u.setBias(u.getBias() + err * learnRate * Params.BIAS);
            u.iterInLinks(k=>{
              k.weight += err* learnRate* k.fromNode.outputValue;
            });
            nnet.addError((setOut[vec][op] - outputs[op]) * (setOut[vec][op] - outputs[op]));
          });

          for(let y=nnet.countLayers() -2; y>0; --y){
            nnet.iterLayer(y,(u,i)=>{
              err=0;
              nnet.iterLayer(2, (o, j)=>{
                err += o.errorValue * o.findLinkIn(u).weight;
              })
              err *= u.outputValue * (1-u.outputValue);

              nnet.iterLayer(y-1,(o,w)=>{
                let k= u.findLinkIn(o);
                k.weight = k.weight + err * learnRate * setIn[vec][w];
              });

              u.setBias(u.getBias() + err * Params.BIAS);
            });
          }
        }
        return nnet.trainedOneCycle();
      }
    }

    if(0){
      let a= new NeuralNet(3,1,{layers:[{size:2, actFunc:"swish"}]});
      let j=a.toJSON();
      let s1,s2;
      console.log(s1=JSON.stringify(j));
      let b= NeuralNet.fromJSON(j);
      let z= b.toJSON();
      console.log("--------------------------------");
      console.log(s2=JSON.stringify(z));
      console.log(`s1==s2 == ${s1==s2}`);

      console.log(
      JSON.stringify(_.groupSimilar([0.57,6.5,0.0007, 0.57, 0.0007,6.5, 4],_.feq)));
    }




    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      NeuralNet, Node, Link, NodeType, FuncType, FuncTypeDB,
      configParams(options){
        return _.inject(Params, options);
      }
    };


    return _$;
  }


  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"))
  }else{
    gscope["io/czlab/mcfud/algo/NNet"]=_module
  }

})(this)


