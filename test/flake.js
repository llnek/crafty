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
  function _module(Mcfud, DQL){

    const Core= Mcfud ? Mcfud["Core"] : gscope["io/czlab/mcfud/core"]();
    const _M = Mcfud ? Mcfud["Math"] : gscope["io/czlab/mcfud/math"]();
    const int=Math.floor;
    const {u:_, is}= Core;

    const {Environment, QLAgent}= DQL;

    /**
     * @module
     */

    const ENV_MAP={
      T_4X4: [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ],
      T_8X8: [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
      ]
    };

    /**
     * @class
     */
    class TestEnv extends Environment{
      #grid;
      #dim;
      #pos;
      constructor(options, N=4){
        super(options);
        _.assert(N==4 || N==8, "bad env size");
        this.#grid=[];
        this.#dim=N;
        ENV_MAP[`T_${N}X${N}`].forEach(r=> r.split("").forEach(c=> this.#grid.push(c)));
        _.assert(this.#grid.length==N*N, `bad grid size ${this.#grid.length}`);
      }
      reset(){
        this.#pos= this.#grid.findIndex(c=> "S");
        return this.#pos;
      }
      actionSpace(){ return ["L","R","U","D"]; }
      #applyAction(action){
        let row= Math.floor(this.#pos / this.#dim);
        let col= this.#pos % this.#dim;
        let v, i=-1, reward=-100, done=0;
        switch(action){
          case "U":
            if(row==0){ reward= -10; }else{
              i= (row-1)*this.#dim + col;
            }
            break;
          case "D":
            if(row== this.#dim-1){ reward= -10; }else{
              i=(row+1)*this.#dim + col;
            }
            break;
          case "L":
            if(col== 0){ reward= -10;}else{
              i=row * this.#dim + col-1;
            }
            break;
          case "R":
            if(col== this.#dim-1) { reward= -10; }else{
              i= row * this.#dim + col+1;
            }
            break;
        }
        if(i>=0){
          v= this.#grid[i];
          this.#pos= i;
          if(v=="G"){
            reward= 999999;
            done=1;
          }else if(v=="H"){
            done=-1;
          }else{
            reward=30;
          }
        }
        return [reward, done];
      }
      getState(){
        return this.#pos;
      }
      step(action){
        const rc = this.#applyAction(action);
        rc.unshift(this.#pos);
        //[new_state, reward, done?]
        return rc;
      }
    }

    class Game{
      constructor(){
      }
      play(options, N){
        console.log(`Starting the Frozen Lake ${N}x${N} Game!!!`);
        let done, reward, env=new TestEnv(options, N);
        let vars=env.getVars();
        let cs, ns, action, agent= new QLAgent(vars.ALPHA,vars.GAMMA,
                                               vars.MIN_EPSILON,vars.MAX_EPSILON,vars.DECAY_RATE);
        let start, eps= vars.EPISODES;
        let W= 1000 * vars.SECS_PER_EPISODE;
        done=false;
        for(let mem, steps, ns,reward, done, cs, i =0; i < eps; ++i){
          cs= env.reset();
          steps=vars.MAX_STEPS;
          mem=[];
          done=0;
          start= _.now();
          //console.log(`Episode ${i} running...`);
          while(steps--){
            if(_.now - start > W){
              //break;
            }
            action= agent.getAction(cs, env.actionSpace());
            [ns, reward, done]= env.step(action);
            agent.updateQValue(cs, action, ns, reward);
            mem.push([cs,action,ns, reward]);
            cs=ns;
            //if(cs > 40) console.log(`new state ===== ${cs}`);
            //if(cs> 50) console.log(`new state ==== ${cs}`);
            if(done != 0){ break; }
          }
          agent.decayEpsilon(i);
          //console.log(mem.reduce((acc,m)=> acc + `[${m[0]}, ${m[1]}, ${m[2]}, ${m[3]}]`, ""));
          if(done >0){
            console.log(`Success!!!! @episode ${i}`);
            console.log(mem.reduce((acc,m)=> acc + `[${m[0]}, ${m[1]}, ${m[2]}, ${m[3]}]`, ""));
            console.log(agent.prnQTable());
            break;
          }
          if(done <0){
            console.log("Failed!!!!");
          }else{
            //console.log(`Finished episode ${i}.`);
          }
        }
        console.log("End Game");
      }
    }

    if(1){
      let o= {
        SECS_PER_EPISODE: 30,
        EPISODES: 250000,
        MAX_STEPS: 450,

        ALPHA: 0.8,
        GAMMA: 0.9,
        MAX_EPSILON: 1.0,
        MIN_EPSILON: 0.001,
        DECAY_RATE: 0.00005
      };
      //new Game().play(o,8);
      new Game().play(o,4);
    }

    return {};
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud"),require("../src/algo/DQL"))
  }else{
    gscope["io/czlab/mcfud/algo/DQL"]=_module
  }
})(this)



