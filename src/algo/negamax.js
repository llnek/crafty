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

  /**Creates the module.
   */
  function _module(Core){

    if(!Core) Core=gscope["io/czlab/mcfud/core"]();
    const {u:_}=Core;

    /**
      * @module mcfud/algo/negamax
      */

    /**
     * @memberof module:mcfud/algo/negamax
     * @class
     * @property {any} lastBestMove
     * @property {any} state
     * @property {any} other
     * @property {any} cur
     */
    class GFrame{
      /**
       * @param {any} cur
       * @param {any} other
       */
      constructor(cur,other){
        this.lastBestMove=UNDEF;
        this.state= UNDEF;
        this.other=other;
        this.cur=cur;
      }
      /**Make a copy of this.
       * @param {function} cp  able to make a copy of state
       * @return {GFrame}
       */
      clone(cp){
        const f= new GFrame();
        f.state=cp(this.state);
        f.lastBestMove=this.lastBestMove;
        f.other=this.other;
        f.cur=this.cur;
        return f;
      }
    }

    /**Represents a game board.
     * @memberof module:mcfud/algo/negamax
     * @class
     */
    class GameBoard{
      constructor(){}
      /**Get the function that copies a game state.
       * @return {function}
       */
      getStateCopier(){}
      /**Get the first move.
       * @param {GFrame} frame
       * @return {any}
       */
      getFirstMove(frame){}
      /**Get the list of next possible moves.
       * @param {GFrame} frame
       * @return {any[]}
       */
      getNextMoves(frame){}
      /**Calculate the score.
       * @param {GFrame} frame
       * @return {number}
       */
      evalScore(frame){}
      /**Check if game is a draw.
       * @param {GFrame} frame
       * @return {boolean}
       */
      isStalemate(frame){}
      /**Check if game is over.
       * @param {GFrame} frame
       * @return {boolean}
       */
      isOver(frame){}
      /**Reverse previous move.
       * @param {GFrame} frame
       * @param {any} move
       */
      unmakeMove(frame, move){
        if(!this.undoMove)
          throw Error("Need Implementation");
        this.switchPlayer(frame);
        this.undoMove(frame, move);
      }
      //undoMove(frame, move){}
      //doMove(frame, move){ }
      /**Make a move.
       * @param {GFrame} frame
       * @param {any} move
       */
      makeMove(frame, move){
        if(!this.doMove)
          throw Error("Need Implementation!");
        this.doMove(frame, move);
        this.switchPlayer(frame);
      }
      /**Switch to the other player.
       * @param {GFrame} frame
       */
      switchPlayer(snap){
        let t = snap.cur;
        snap.cur= snap.other;
        snap.other= t;
      }
      /**Get the other player.
       * @param {any} pv player
       * @return {any}
       */
      getOtherPlayer(pv){
        if(pv === this.actors[1]) return this.actors[2];
        if(pv === this.actors[2]) return this.actors[1];
      }
      /**Get the current player.
       * @return {any}
       */
      getPlayer(){
        return this.actors[0]
      }
      /**Take a snapshot of current game state.
       * @return {GFrame}
       */
      takeGFrame(){}
      run(seed,actor){
        this.getAlgoActor=()=>{ return actor }
        this.syncState(seed,actor);
        let pos= this.getFirstMove();
        if(_.nichts(pos))
          pos= _$.evalNegaMax(this);
        return pos;
      }
    }

    /** @ignore */
    function _calcScore(board,game,depth,maxDepth){
      //if the other player wins, then return a -ve else +ve
      //maxer == 1 , minus == -1
      let score=board.evalScore(game,depth,maxDepth);
      /*
      if(!_.feq0(score))
        score -= 0.01*depth*Math.abs(score)/score;
      return score;
      */
      return score * (1 + 0.001 * depth);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //option2
    function _negaAlphaBeta(board, game, depth, maxDepth, alpha, beta){

      if(depth==0 || board.isOver(game)){
        return { depth, value: _calcScore(board,game,depth,maxDepth) }
      }

      let state=game,
          copier= board.getStateCopier(),
          openMoves= _.shuffle(board.getNextMoves(game));

      for(let rc, move, i=0; i< openMoves.length; ++i){
        move= openMoves[i];
        if(!board.undoMove){
          _.assert(copier, "Missing state copier!");
          game= state.clone(copier);
        }
        board.makeMove(game, move);
        rc = _negaAlphaBeta(board, game, depth-1,
                                         maxDepth,
                                         {value: -beta.value, move: beta.move},
                                         {value: -alpha.value, move: alpha.move});
        //now, roll it back
        if(board.undoMove)
          board.unmakeMove(game, move);
        rc.value = -rc.value;
        rc.move = move;
        if(rc.value>alpha.value){
          alpha = {value: rc.value, move: move, depth: rc.depth};
        }
        if(alpha.value >= beta.value){
          return beta;
        }
      }

      return JSON.parse(JSON.stringify(alpha));
    }

    /**Implements the NegaMax Min-Max algo.
     * @see {@link https://github.com/Zulko/easyAI}
     * @param {GameBoard} board
     * @param {GFrame} game
     * @param {number} depth
     * @param {number} maxDepth
     * @param {number} alpha
     * @param {number} beta
     * @return {number}
     */
    function _negaMax(board, game, depth,maxDepth,alpha, beta){

      if(depth==0 || board.isOver(game)){
        return [_calcScore(board,game,depth,maxDepth),null]
      }

      let openMoves = _.shuffle(board.getNextMoves(game)),
          copier= board.getStateCopier(),
          state=game,
          bestValue = -Infinity,
          bestMove = openMoves[0];

      if(depth==maxDepth)
        state.lastBestMove=bestMove;

      for(let rc, move, i=0; i<openMoves.length; ++i){
        if(!board.undoMove){
          _.assert(copier, "Missing state copier!");
          game=state.clone(copier);
        }
        move = openMoves[i];
        //try a move
        board.makeMove(game, move);
        rc= - _negaMax(board, game, depth-1, maxDepth, -beta, -alpha)[0];
        //now, roll it back
        if(board.undoMove)
          board.unmakeMove(game, move);
        //how did we do ?
        if(bestValue < rc){
          bestValue = rc;
          bestMove = move
        }
        if(alpha < rc){
          alpha=rc;
          if(depth == maxDepth)
            state.lastBestMove = move;
          if(alpha >= beta) break;
        }
      }

      return [bestValue, state.lastBestMove];
    }

    const _$={
      algo:"negamax",
      GFrame,
      GameBoard,
      /**Make a move on the game-board using negamax algo.
       * @memberof module:mcfud/algo/negamax
       * @param {GameBoard} board
       * @return {any} next best move
       */
      XXevalNegaMax(board){
        const f= board.takeGFrame();
        const d= board.depth;
        let [score,move]= _negaMax(board, f, d,d, -Infinity, Infinity);
        if(_.nichts(move))
          console.log(`evalNegaMax: score=${score}, pos= ${move}, lastBestMove=${move}`);
        return move;
      },
      evalNegaMax(board){
        const f= board.takeGFrame();
        const d= board.depth;
        let {value, move} = _negaAlphaBeta(board, f, d, d, {value: -Infinity },
                                                           {value: Infinity  });
        if(_.nichts(move))
          console.log(`evalNegaMax: score= ${value}, pos= ${move}`);
        return move;
      }
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("@czlab/mcfud")["Core"])
  }else{
    gscope["io/czlab/mcfud/algo/negamax"]=_module
  }

})(this);

