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


;(function(UNDEF){

  "use strict";


  /**Create the module.
  */
  function _module(So,Se,Gr,Mz,Ge,Min,Neg,Dql,N_C,N_B){
    return {
      Sort:So,
      Search:Se,
      Graph:Gr,
      Maze:Mz,
      Genetic:Ge,
      Minimax:Min,
      Negamax:Neg,
      DQL:Dql,
      NNet:N_C,
      NEAT:N_B
    }
  }

  //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  //exports
  if(typeof module=="object" && module.exports){
    module.exports=_module(require("algo/sort.js"),
      require("algo/search.js"),
      require("algo/graph.js"),
      require("algo/maze.js"),
      require("algo/genetic.js"),
      require("algo/minimax.js"),
      require("algo/negamax.js"),
      require("algo/DQL.js"),
      require("algo/nnet.js"),
      require("algo/NEAT.js")
    );
  }else{
  }

})(this);

