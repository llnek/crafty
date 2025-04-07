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
     * @module mcfud/algo/DQL
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const Params={

      ALPHA: 0.1,
      GAMMA: 0.95,
      MAX_EPSILON: 1.0,
      DECAY_RATE: 0.001,
      MIN_EPSILON: 0.05,

      MAX_STEPS: 250,
      EPISODES: 1000,
      SECS_PER_EPISODE: 30

    };

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    function argMax(arr){
      let max= -Infinity, pos= -1;
      arr.forEach((v,i)=>{
        if(v>max){
          max=v;pos=i;
        }
      });
      return [pos, max];
    }

    /**
     * @class
     */
    class QLAgent{
      #maxEpsilon;
      #minEpsilon;
      #alpha;
      #gamma;
      #decayRate;
      #qtable;
      constructor(alpha,gamma,minEpsilon,maxEpsilon,decayRate){
        this.#maxEpsilon=maxEpsilon;
        this.#minEpsilon=minEpsilon;
        this.#decayRate=decayRate;
        this.#alpha=alpha;
        this.#gamma=gamma;
        this.#qtable = new Map();
      }
      #decodeKey(state){
        return (is.str(state) ||
          is.num(state) ||
          is.bool(state)) ? state : JSON.stringify(state);
      }
      #safeGetState(state){
        let key=this.#decodeKey(state);
        if(!this.#qtable.has(key))
          this.#qtable.set(key, new Map());
        return this.#qtable.get(key);
      }
      #safeGetAction(state,action,dft=0){
        let s= this.#safeGetState(state);
        return s.has(action) ? s.get(action) : dft;
      }
      getQValue(state, action){
        return this.#safeGetAction(state,action);
      }
      updateQValue(state, action, nextState, reward){
        let cv = this.getQValue(state,action);
        let m= this.#safeGetState(nextState);
        let ks=m.keys().toArray().sort();
        let nvs= ks.map(a=> this.getQValue(nextState,a));
        let max = nvs.length>0 ? argMax(nvs)[1] : 0;
        // q-learning formula
        let nv= cv + this.#alpha * (reward + this.#gamma * max - cv);
        this.#safeGetState(state).set(action, nv);
      }
      getAction(state, listOfActions){
        if(_.rand() < this.#maxEpsilon)
          return _.randItem(listOfActions);
        //choose action with highest q-value
        let max= -Infinity,
            rs= listOfActions.reduce((acc, a, i)=> {
              i= this.getQValue(state,a);
              acc.push([a, i]);
              if(i> max){ max=i }
              return acc;
            },[]);
        let choices= rs.filter(a=> a[1] == max);
        return _.randItem(choices)[0];
      }
      decayEpsilon(episode){
        this.#maxEpsilon = this.#minEpsilon + (this.#maxEpsilon - this.#minEpsilon) * Math.exp(-this.#decayRate *episode);
        //this.#maxEpsilon = Math.max(0, this.#maxEpsilon - this.#decayRate);
      }
      prnQTableAsObj(){
        let obj={};
        this.#qtable.keys().toArray().sort().forEach(k=>{
          let v, o={}, m= this.#qtable.get(k);
          obj[k]=o;
          m.keys().toArray().sort().forEach(k=>{
            o[k]= m.get(k)
          });
        });
        return JSON.stringify(obj);
      }
      prnQTable(){
        let obj=[];
        this.#qtable.keys().toArray().sort().forEach(k=>{
          let v, m= this.#qtable.get(k);
          m.keys().toArray().sort().forEach(i=>{
            v = m.get(i);
            obj.push(`${k},${i},${v}`);
          });
        });
        return obj.join("\n");
      }
      save(){
        //save state,action,qvalue
        //save to file system...
        return this.prnQTable();
      }
      load(data){
        let m, r, arr= data.split("\n");
        m= new Map();
        arr.forEach(a=>{
          r= a.split(",");
          if(!m.has(r[0]))
            m.set(r[0], new Map());
          n= m.get(r[0]);
          n.set(r[1], r[2]);
        });
        this.#qtable=m;
      }
    }

    /**
     * @class
     */
    class Environment{
      #vars;
      constructor(options){
        this.#vars= Object.freeze(_.inject({}, Params, options));
      }
      getVars(){ return this.#vars }
      reset(){
        _.assert(false, "Please implement reset()");
      }
      actionSpace(){
        _.assert(false, "Please implement actionSpace()");
      }
      getState(){
        _.assert(false, "Please implement getState()");
      }
      step(action){
        _.assert(false, "Please implement step()");
      }
    }

    const _$={
      Environment,
      QLAgent
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"));
  }else{
    gscope["io/czlab/mcfud/algo/DQL"]=_module
  }

})(this)



