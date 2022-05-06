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
// Copyright © 2013-2022, Kenneth Leung. All rights reserved.

;(function(gscope,UNDEF){

  "use strict";

  /**Create the module.
   */
  function _module(Core,Basic,Sort){

    if(!Core) Core= gscope["io/czlab/mcfud/core"]();
    if(!Basic) Basic= gscope["io/czlab/mcfud/algo/basic"]();
    if(!Sort) Sort= gscope["io/czlab/mcfud/algo/sort"]();

    const {Bag,Stack,Queue,StdCompare:CMP,prnIter}= Basic;
    const {MinPQ}= Sort;
    const int=Math.floor;
    const {is,u:_}= Core;

    /**
     * @module mcfud/algo_prog
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Fibonacci{
      static solve0(N){
        _.assert(N>0,"bad fib number");
        let res=[];
        function _s(n){
          return n<3?1: (_s(n-1) + _s(n-2))
        }
        for(let i=1;i<=N;++i){ res.push(_s(i)) }
        return res;
      }
      static solve(N){
        _.assert(N>0,"bad fib number");
        let res=[],
            c=new Map();
        function _s(n){
          if(n<3) return 1;
          if(c.has(n)) return c.get(n);
          let v= _s(n-2) + _s(n-1);
          c.set(n,v);
          return v;
        }
        for(let i=1;i<=N;++i){ res.push(_s(i)) }
        return res;
      }
      static test(){
        let vs=Fibonacci.solve(15);
        console.log(vs.join(","));
      }
    }
    //Fibonacci.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Classic NQueen problem.
     * @memberof module:mcfud/algo_prog
     * @class
     */
    class NQueen{
      /**Place N chess queens on an N×N chessboard so that
       * no two queens can attack each other.
       * @param {number} size board size
       * @return {array} sizexsize board
       */
      static solve(N){
        //_.assert(N>3,"Bad board for NQueen: " + N);
        let M='1',
            Z='0',
            G=_.fill(N,()=> _.fill(N,Z));
        function _qok(row,col){
          for(let x=0; x<col; ++x) if(G[row][x] == M) return false;
          for(let y=row, x=col; y>=0 && x>=0; --y, --x) if(G[y][x] == M) return false;
          for(let y=row, x=col; x>=0 && y<N; ++y, --x) if(G[y][x] == M) return false;
          return true;
        }
        //start from left to right
        return (function _s(col){
          for(let y=0;y<N;++y){
            if(_qok(y,col)){
              G[y][col] = M;
              if(col+1>=N || _s(col+1))
              return G;
              //else revert
              G[y][col] = Z;
            }
          }
        })(0);
      }
      static test(){
        let g=NQueen.solve(8);
        if(g)
          g.forEach(row=> console.log(row.join(",")));
        else
          console.log("no solution");
      }
    }
    //NQueen.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Classic Kinight's tour problem using Warnsdorff's algorithm.
     * @memberof module:mcfud/algo_prog
     * @class
     */
    class KnightsTour{
      static N=8;
      static solve(sy,sx){
        let N=KnightsTour.N,
            ZZ=N*N,
            G= _.fill(N,()=> _.fill(N, 0)),
            //dirs=[[1,2],[1,-2],[2,1],[2,-1],[-1,2],[-1,-2],[-2,1],[-2,-1]];
            //dirs=[[1,2], [2,1], [2,-1], [1,-2], [-1,-2], [-2,-1], [-2,1], [-1,2 ]];
            dirs=[ [2,1],[1,2],[-1,2],[-2,1],[-2,-1],[-1,-2],[1,-2],[2,-1] ];

        function _ok(y,x){
          return x>=0 && x<N && y>=0 && y<N && G[y][x]<0 }
        function _getAdj(row,col,cnt=0){
          for(let k=0; k<N; ++k)
            if(_ok(row+dirs[k][1], col+dirs[k][0])) ++cnt;
          return cnt;
        }
        function _validAdj(row,col,y,x){
          for(let k=0; k<N; ++k)
            if(y+dirs[k][1] == row &&
               x+dirs[k][0] == col) return true;
        }
        function _move(ret){
          let minDeg = N+1,
              minDegIdx = -1,
              [col,row] = ret,
              c,nx,ny,start = _.randInt(N);
          for(let c,i,k=0; k < N; ++k){
            i = (k+start) % N;
            ny = row+dirs[i][1];
            nx = col+dirs[i][0];
            if(_ok(ny,nx)){
              if((c= _getAdj(ny,nx))<minDeg){
                minDegIdx = i;
                minDeg = c;
              }
            }
          }
          if(minDegIdx <0){
            ret=null; // could not find a next cell
          }else{
            ny= row+ dirs[minDegIdx][1];
            nx= col+ dirs[minDegIdx][0];
            G[ny][nx] = G[row][col]+1;
            // update next point
            ret[0]=nx;
            ret[1]=ny;
          }
          return ret;
        }
        function _solve(row,col){
          // using Warnsdorff's heuristic
          let ret=[col,row];
          G[row][col]=1;
          for(let i=0; i< ZZ-1; ++i){
            ret = _move(ret);
            if(!ret) return false;
          }
          // Check if tour is closed (Can end at starting point)
          if(!_validAdj(row,col,ret[1],ret[0])) return false;
          return G;
        }
        while(!_solve(sy,sx))
          G.forEach(r=> r.forEach((v,i)=> r[i]= -1));
        return G;
      }
      static test(){
        let g=KnightsTour.solve(_.randInt(KnightsTour.N),_.randInt(KnightsTour.N));
        if(g)
          g.forEach(row=> console.log(row.map(v=> _.prettyNumber(v,2," ")).join(",")));
        else
          console.log("no solution");
      }
    }
    //KnightsTour.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Classic Sudoku.
     * @memberof module:mcfud/algo_prog
     * @class
     */
    class Sudoku{
      /**Solve the given sudoku.
       * @param {array} G grid NxN(N=9)
       * @return {array} the solution
       */
      static solve(G){
        const N=G.length,
              W=G[0].length;
        _.assert(N==W,"bad sudoku size");
        function _qok(row, col, v){
          for(let c=0; c < G[0].length; ++c) if(G[row][c] == v) return false;//check row
          for(let r=0; r < G.length; ++r) if(G[r][col] == v) return false;//check col
          //check box
          let boxX= col - col % 3,
              boxY= row - row % 3;
          for(let r=boxY; r<boxY+3; ++r)
          for(let c = boxX; c<boxX+3; ++c){
            if(G[r][c] == v) return false;
          }
          return true;
        }
        return (function _s(){
          let row,col;
          //pick first empty slot
          for(let r=0; r<N; ++r){
            for(let c=0; c<N; ++c){
              if(G[r][c]==0){
                row = r;
                col = c;
                break;
              }
            }
            if(row !==undefined){ break }
          }
          if(row===undefined){//all filled
            return G
          }else{//test or backtrack
            for(let v=1; v <= N; ++v){
              if(_qok(row, col, v)){
                G[row][col] = v;
                if(_s()){
                  return G
                }else{//revert
                  G[row][col] = 0;
                }
              }
            }
          }
        })();
      }
      static test(){
        let D=`3 0 6 5 0 8 4 0 0
               5 2 0 0 0 0 0 0 0
               0 8 7 0 0 0 0 3 1
               0 0 3 0 1 0 0 8 0
               9 0 0 8 6 3 0 0 5
               0 5 0 0 9 0 6 0 0
               1 3 0 0 0 0 2 5 0
               0 0 0 0 0 0 0 7 4
               0 0 5 2 0 6 3 0 0`.split(/\s+/).map(n=> {return +n});
        let g= Sudoku.solve(_.partition(9,D));
        if(g)
          g.forEach(r=> console.log(r.join(",")));
        else
          console.log("no solution");
      }
    }
    //Sudoku.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Classic coin change.
     * @memberof module:mcfud/algo_prog
     * @class
     */
    class UnboundedKnapsack{
      /**Count the number of ways coins can be combined to
       * match the given amount.
       * @param {array} coins list of coins
       * @param {number} amount
       * @return {number} number of solutions
       */
      static solve(coins,amount){
        return (function _s(s,m,n){
          // If n is 0 then there is 1 solution (do not include any coin)
          if(n == 0) return 1;
          // If n is less than 0 then no solution exists
          if(n < 0) return 0;
          // If there are no coins and n is greater than 0, then no solution exist
          if(m <=0 && n >= 1) return 0;
          // count is sum of solutions (i) including S[m-1] (ii) excluding S[m-1]
          return _s(s, m-1, n) +
                 _s(s, m, n - s[m-1]);
        })(coins,coins.length,amount);
      }
      /**Count the number of ways coins can be combined to
       * match the given amount.
       * @param {array} coins list of coins
       * @param {number} amount
       * @return {number} number of solutions
       */
      static solve2(coins,amount,V){
        const N=coins.length;
        return (function _s(s,v,n,dp){
          if(v == 0) return dp[n][v] = 1;
          if(n == 0) return 0;
          if(dp[n][v] != -1) return dp[n][v];
          if(s[n-1] <= v){
            // Either Pick this coin or not
            return dp[n][v] = _s(s, v- s[n-1], n, dp) + _s(s, v, n-1, dp);
          }else{ // We have no option but to leave this coin
            return dp[n][v] = _s(s, v, n-1, dp);
          }
        })(coins,amount,N, V);
      }
      static test(){
        let ret=UnboundedKnapsack.solve([1,2,3],4);
        console.log("n# of solution= "+ ret);
      }
      static test2(){
        let amt=6,N=3,V=_.fill(N+1,()=> _.fill(amt+1,-1));
        let ret=UnboundedKnapsack.solve2([1,2,3], amt,V);
        console.log("n# of solution= "+ ret);
      }
    }
    //UnboundedKnapsack.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Classic min coins.
     * @memberof module:mcfud/algo_prog
     * @class
     */
    class MinCoins{
      static solve(coins,amount){
        return (function _s(coins,m,V){
          // table[i] will be storing the minimum number of coins
          // required for i value. So table[V] will have result
          let table = _.fill(V+1, Infinity);
          table[0] = 0; // base case (V is 0)
          // minimum coins required for all values from 1 to V
          for(let v=1; v<=V; ++v){
            // Go through all coins smaller than i
            for(let r,j=0; j<m; ++j)
              if(coins[j] <= v){
                r = table[v - coins[j]];
                if(r !== Infinity && r+1 < table[v]) table[v] = r+1;
              }
          }
          return table[V]==Infinity? -1: table[V];
        })(coins,coins.length,amount);
      }
      static test(){
        let coins = [9, 6, 5, 1];
        let amt = 11;
        console.log("Minimum coins required is " + MinCoins.solve(coins, amt));
      }
    }
    //MinCoins.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    /**Classic min coins.
     * @memberof module:mcfud/algo_prog
     * @class
     */
    class MinCoins2{
      static solve(coins,amount){
        function _f(s,m,V,dp,out){
          if(V == 0){ return out }
          for(let i=0; i<m; ++i){
            // Try every coin that has value smaller than n
            if(V-s[i] >= 0 && dp[V-s[i]]+1 == dp[V]){
              out.push(s[i]);
              _f(s,m,V-s[i],dp,out);
              break;
            }
          }
          return out;
        }
        function _s(s,m,V,dp){
          if(V == 0){ return (dp[0]=0); }//amount=0
          //previously computed subproblem occurred
          if(dp[V] != -1) return dp[V];
          // try every coin that has smaller value than V
          let ret = Infinity;
          for(let x,i=0; i<m; ++i){
            if(s[i] <= V){
              x = _s(s,m,V-s[i],dp);
              if(x !== Infinity)
                ret = Math.min(ret, 1+x);
            }
          }
          // Memoizing value of current state
          return (dp[V] = ret);
        }
        return (function(s, m, V){
          let out=[],
              dp= _.fill(V+1,-1);
          if(_s(s,m,V,dp) !== Infinity){
            return _f(s,m,V,dp,out)
          }
        })(coins, coins.length, amount);
      }
      static test(){
        let g=MinCoins2.solve([2,3,4],5);//[2,3,4,5],21);
        if(g)
          console.log(g.toString());
        else
          console.log("no solution");
      }
    }
    //MinCoins2.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class TowerOfHanoi{
      static solve(N){
        let steps=[],
            C=0,
            d=new Stack(),
            t=new Stack(),
            s=new Stack();
        s.name="P1";
        t.name="P2";
        d.name="P3";
        for(let i=N;i>0;--i)s.push(i);
        (function _s(src,des,tmp,n){
          if(n<=0){return}
          _s(src, tmp, des, n-1);
          let d= src.pop();
          des.push(d);
          steps.push(`move d${d} from ${src.name} to ${des.name}`);
          _s(tmp,des,src,n-1);
        })(s,d,t,N);
        _.assert(s.isEmpty()&&t.isEmpty()&&d.size()==N,"hanoi has problems");
        for(let it=d.iter();it.hasNext();){
          ++C;
          _.assert(C== it.next(), "Hanoi has problems");
        }
        _.assert(C==N,"hanoi has prolems");
        return steps;
      }
      static test(){
        console.log(TowerOfHanoi.solve(4).join("\n"));
      }
    }
    //TowerOfHanoi.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class GenerateGuess{
      static solve(target){
        const GS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.";
        const GS2= GS.split("");
        const LEN= target.length;
        //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        function genGuess(genes){
          while(genes.length < LEN){
            let n = Math.min(LEN-genes.length, GS.length);
            _.shuffle(GS2);
            for(let i=0;i<n;++i) genes.push(GS2[i]);
          }
          return genes.join("");
        }
        function getFitness(guess){
          let sum=0;
          for(let i=0;i<LEN;++i)
            if(target.charAt(i) == guess.charAt(i)) ++sum;
          return sum;
        }
        function mutate(parent){
          let c = parent.split(""),
              i = _.randInt(parent.length);
          _.shuffle(GS2);
          c[i]= GS2[0] == c[i] ? GS2[GS2.length-1] : GS2[0];
          return c.join("");
        }
        function show(guess){
          let f= getFitness(guess);
          console.log(`fitness=${f} for guess ${guess}`);
        }
        let bestPar= genGuess([]);
        let bestFit= getFitness(bestPar);
        while(1){
          let c= mutate(bestPar);
          let f= getFitness(c);
          if(bestFit >= f) continue;
          show(c);
          if(f >= bestPar.length)break;
          bestFit= f;
          bestPar= c;
        }
      }
    }
    //GenerateGuess.solve("Hello World!");

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class Tiles{
      static Node(data,level,fval){
        return{
          data,level,fval,
          genChild(){
            //Generate child nodes from the given node by moving the blank space
            //either in the four directions {up,down,left,right}
            let [x,y] = this.find(this.data,0);
            //val_list contains position values for moving the blank space in either of
            //the 4 directions [up,down,left,right] respectively.
            let vlist = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]];
            let c,children = [];
            vlist.forEach(v=>{
              c= this.shuffle(this.data,x,y,v[0],v[1]);
              if(c)
                children.push(Tiles.Node(c,this.level+1,0))
            });
            return children;
          },
          shuffle(puz,x1,y1,x2,y2){
            //Move the blank space in the given direction and if the position value are out
            //of limits the return None
            if(x2 >= 0 && x2 < this.data.length && y2 >= 0 && y2 < this.data.length){
              let t,temp_puz = [];
              temp_puz = this.copy(puz);
              t= temp_puz[x2][y2];
              temp_puz[x2][y2] = temp_puz[x1][y1];
              temp_puz[x1][y1] = t;
              return temp_puz;
            }
          },
          copy(root){
            //Copy function to create a similar matrix of the given node
            return _.deepCopyArray(root);
          },
          find(puz,x){
            //Specifically used to find the position of the blank space
            for(let i=0;i<this.data.length;++i)
              for(let j=0;j<this.data.length;++j)
                if(puz[i][j] == x) return [i,j];
          }
        }
      }
      static Puzzle(size){
        return{
          n: size, open: [], closed: [],
          f(start,goal){
            //Heuristic Function to calculate hueristic value f(x) = h(x) + g(x)
            return this.h(start.data,goal)+start.level;
          },
          h(start,goal){
            //Calculates the different between the given puzzles
            let temp = 0;
            for(let i=0;i<size;++i)
              for(let j=0;j<size;++j)
                if(start[i][j] != goal[i][j] && start[i][j] != 0) temp += 1;
            return temp;
          },
          process(init,goal){
            //Accept Start and Goal Puzzle state
            let cur,start = Tiles.Node(init,0,0);
            start.fval = this.f(start,goal);
            //Put the start node in the open list
            this.open.push(start);
            console.log("\n\n");
            while(1){
              cur = this.open[0];
              console.log("");
              console.log(" \\\'/ \n");
              cur.data.forEach(i=> console.log(i.join(" ")));
              //If the difference between current and goal node is 0 we have reached the goal node
              if(this.h(cur.data,goal) == 0) break;
              cur.genChild().forEach(i=>{
                i.fval = this.f(i,goal);
                this.open.push(i);
              });
              this.closed.push(cur);
              this.open.shift();
              //sort the opne list based on f value
              this.open.sort((a,b)=>{
                return a.fval<b.fval?-1:(a.fval>b.fval?1:0)
              });
            }
          }
        }
      }
      static solve(init,goal){
        Tiles.Puzzle(3).process(init,goal);
      }
    }
    //Tiles.solve([[1,2,3],[0,4,6],[7,5,8]],[[1,2,3],[4,5,6],[7,8,0]]);
    //Tiles.solve([[2,5,3],[1,6,0],[7,8,4]],[[0,1,2],[3,4,5],[6,7,8]]);

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class BBTiles{
      static N=3;
      static Node(parent, mat, emptyTilePos, cost, level){
        return{
          parent, mat, emptyTilePos, cost, level
        }
      }
      static calcCost(mat, final){
        let n=BBTiles.N,c= 0;
        for(let i=0;i<n;++i)
          for(let j=0;j<n;++j)
            if(mat[i][j] != final[i][j]) c+=1;
        return c;
      }
      static newNode(mat, empty_tile_pos, new_empty_tile_pos, level, parent, final){
        let new_mat = _.deepCopyArray(mat);
        //Move tile by 1 position
        let x1 = empty_tile_pos[0];
        let y1 = empty_tile_pos[1];
        let x2 = new_empty_tile_pos[0];
        let y2 = new_empty_tile_pos[1];
        let tmp= new_mat[x1][y1];

        new_mat[x1][y1] = new_mat[x2][y2];
        new_mat[x2][y2] = tmp;

        // Set number of misplaced tiles
        let cost = BBTiles.calcCost(new_mat, final);
        return BBTiles.Node(parent, new_mat, new_empty_tile_pos, cost, level);
      }
      static isSafe(x, y){
        return x >= 0 && x < BBTiles.N && y >= 0 && y < BBTiles.N;
      }
      static printMatrix(mat){
        console.log("");
        console.log(" \\\'/ \n");
        for(let i=0;i<BBTiles.N;++i)
          console.log(mat[i].join(" "));
      }
      static printPath(root){
        if(root){
          BBTiles.printPath(root.parent);
          BBTiles.printMatrix(root.mat);
        }
      }
      static solve(init, final, zero){
        //# bottom, left, top, right
        const row = [ 1, 0, -1, 0 ];
        const col = [ 0, -1, 0, 1 ];
        let pq = new MinPQ((a,b)=>{
          return a.cost<b.cost?-1:(a.cost>b.cost?1:0)
        });
        let cost = BBTiles.calcCost(init, final);
        let root = BBTiles.Node(null, init, zero, cost, 0);
        pq.insert(root);
        while(!pq.isEmpty()){
          let minimum = pq.delMin();
          if(minimum.cost == 0){
            BBTiles.printPath(minimum);
            return;
          }
          for(let i=0;i<BBTiles.N;++i){
            let c,new_tile_pos = [minimum.emptyTilePos[0] + row[i],
                                  minimum.emptyTilePos[1] + col[i] ];
            if(BBTiles.isSafe(new_tile_pos[0], new_tile_pos[1])){
              c= BBTiles.newNode(minimum.mat,
                                 minimum.emptyTilePos,
                                 new_tile_pos, minimum.level + 1, minimum, final);
              pq.insert(c);
            }
          }
        }
      }
    }
    //BBTiles.solve([[1,2,3],[5,6,0],[7,8,4]], [[1,2,3],[5,8,6],[0,7,4]],[1,2]);


    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class TwoSum{
      static solve4(input, sum){
        // Time: O(N) Space: O(1)
        let i=0, j =input.length - 1;
        input.sort();
        while(i < j){
          let sigma = input[i] + input[j]
          if(sigma == sum) return true;
          if(sigma > sum)
            --j;
          else
            ++i;
        }
      }
      static solve3(input, sum){
        // Time: O(N) Space: O(N)
        let m = new Map();
        for(let i=0;i<input.length;++i){
          let other=sum - input[i];
          if(m.has(other))
            return [i, m.get(other)]
          m.set(input[i], i);
        }
      }
      static solve2(input, sum){
        // Time: O(N) Space: O(N)
        let bin=new Set(input);
        for(let i=0; i< input.length;++i){
          if(input[i] != sum &&
             bin.has(Math.abs(sum-input[i]))) return true;
        }
      }
      static solve1(input, sum){
        //Time: O(N^2),Space: O(1)
        for(let i=0;i< input.length;++i){
          for(let j=i+1; j < input.length;++j){
            if(input[j] == (sum - input[i])) return true
          }
        }
      }
      static test(){
        let vs=TwoSum.solve1([3,1,5,2,9], 7);
        console.log("ans= " + vs);
      }
    }
    //TwoSum.test();

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    class TenAlgos{
      static solve1(){
        //Find the missing number in the array
        function find(input){
          let n = input.length + 1;
          let sum = 0;
          for(let i in input){
            sum += input[i];
          }
          return Math.floor((n * (n + 1)) / 2) - sum;
        };
        return find([1,2,3,4,6,7,8,9],10)
      }
      static solve2(){
        //inverting ints
        function reverse(num){
          let result = 0;
          while(num != 0){
            result = result * 10 + num % 10;
            num = Math.trunc(num / 10);
          }
          if (result > 2**31 || result < -(2**31)) return 0;
          return result;
        }
        return reverse(-123)
      }
      static solve3(){
        //permute
        function permute(nums){
          let results = [];
          let go = (current) => {
            if (current.length == nums.length){
              results.push(current);
              return;
            }
            nums.forEach(n => {
              if(!current.includes(n)){
                go([...current, n]);
              }
            });
          }
          go([]);
          return results;
        }
        return permute([1,2,3])
      }
      static solve4(){
        function check(s1, s2){
          const
            len1 = s1.length,
            len2 = s2.length;
          if(len1 > len2) return false;
          const count = Array(26).fill(0);
          for(let i = 0; i < len1; ++i){
            count[s1.charCodeAt(i)-97]++;
            count[s2.charCodeAt(i)-97]--;
          }
          if(!count.some(e => e != 0)) return true;
          for(let i = len1; i < len2; ++i){
            count[s2.charCodeAt(i)-97]--;
            count[s2.charCodeAt(i-len1)-97]++;
            if(!count.some(e => e != 0)) return true;
          }
          return false;
        }
        //return check("ab", "eidbao");
        return check("aa", "eidbao");
      }
      static solve5(){
        function longestValid(S){
          let stack = [-1], ans = 0;
          for(let i = 0; i < S.length; ++i){
            if(S[i] == "("){
              stack.push(i);
            }else if(stack.length == 1){
              stack[0] = i;
            }else{
              stack.pop();
              ans = Math.max(ans, i - stack[stack.length-1]);
            }
          }
          return ans
        }
        //return longestValid("(()");
        return longestValid(")()())");
      }
      static solve6(){
        //4 sum
        function fourSum(nums, target){
          let result = [];
          let length = nums.length;
          if(length < 4) return result;
          nums.sort((a, b) => a - b );
          for(let i = 0; i < length - 3; ++i){
            if(nums[i] == nums[i - 1]) continue;
            for(let j = i + 1; j < length - 2; ++j){
              if(j > i + 1 && nums[j] == nums[j - 1]) continue;
              let k = j + 1;
              let l = length - 1;
              while (k < l) {
                const sum = nums[i] + nums[j] + nums[k] + nums[l];
                if(sum == target){
                  result.push([nums[i], nums[j], nums[k], nums[l]])
                }
                if(sum <= target){
                  k += 1;
                  while(nums[k] == nums[k - 1]) {
                    k += 1;
                  }
                }
                if (sum >= target){
                  l -= 1;
                  while(nums[l] == nums[l + 1]) {
                    l -= 1;
                  }
                }
              }
            }
          }
          return result;
        }

        return fourSum([1,4,5,6,8,3,2], 12)
      }
      static solve7(){
        //string mult
        function mult(num1, num2){
         if(num1 == 0 || num2 == 0) return "0";
         const result = [];
          for(let a = num1.length -1; a >= 0; --a){
            for(let b = num2.length - 1; b >= 0; --b){
              const p1 = a + b;
              const p2 = a + b + 1;
              const sum = (result[p2] ?? 0) + num1[a] * num2[b];
              result[p1] = (result[p1] ?? 0) + Math.floor(sum / 10);
              result[p2] = sum % 10;
            }
          }
          result[0] == 0 && result.shift();
          return result.join("");
        }
        return mult("5", "4");
      }
      static solve8(){
        function shortestPalindrome(s){
          let index = 0;
          for(let i = s.length - 1; i >= 0; --i){
            if(s[i] == s[index]) index++;
          }
          if(index == s.length) return s;
          let remainingRev = s.substring(index, s.length);
          remainingRev = reverse(remainingRev);
          return remainingRev + shortestPalindrome(s.substring(0, index)) + s.substring(index);
        }
        function reverse(s){
          let out = "";
          for(let i = s.length - 1; i >= 0; --i){
            out += s[i];
          }
          return out;
        }

        //return shortestPalindrome("aacecaaa");
        return shortestPalindrome("abcd");
      }
      static solve9(){
        function numberToWords(num){
          let result = toHundreds(num % 1000);
          const bigNumbers = ["Thousand", "Million", "Billion"];
          for(let i = 0; i < 3; ++i){
            num = Math.trunc(num / 1000);
            result = num % 1000 != 0 ?
              [toHundreds(num % 1000), bigNumbers[i], result].filter(Boolean).join(" ") : result;
          }
          return result.length == 0 ? "Zero" : result;
        }
        function toHundreds(num){
          const numbers = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"];
          const tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"];
          const result = Array(3).fill("");
          let a = Math.trunc(num / 100), b = num % 100, c = num % 10;
          result[0] = a > 0 && `${numbers[a]} Hundred`;
          result[1] = b < 20 ? numbers[b] : tens[Math.trunc(b / 10)]
          result[2] = b >= 20 && `${numbers[c]}`;
          return result.filter(Boolean).join(" ");
        }
        return numberToWords(17843);
      }
      static solve10(){
        function canConstruct(ransomNote, magazine){
         if(ransomNote.length > magazine.length) return false;
         let magMap = new Map();
          for(let char of magazine){
            magMap.set(char, (magMap.get(char) || 0 ) + 1);
          }
          for(let note of ransomNote){
            let counter = magMap.get(note);
            if(!counter) return false;
            magMap.set(note, --counter);
          }
          return true;
        }
        return canConstruct("aa", "aab")
      }
      static test(){
        let vs=TenAlgos.solve10();
        console.log("ans= " + vs);
      }
    }
    //TenAlgos.test();




    return {}
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("mcfud"),require("../src/algo/basic"),require("../src/algo/sort"))
  }else{
    gscope["io/czlab/mcfud/algo/prog"]=_module
  }

})(this);


