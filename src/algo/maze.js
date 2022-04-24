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
  function _module(Core,_M){

    if(!Core) Core= gscope["io/czlab/mcfud/core"]();
    if(!_M) _M= gscope["io/czlab/mcfud/math"]();

    const {is,u:_}= Core;

    /**
     * @module mcfud/algo_maze
     */

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    // Replace a character at index in a string
    function replaceAt(s, i, r){
      return (i> s.length-1)?s :`${s.substr(0,i)}${r}${s.substr(i+1)}`
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    function stringVal(s, i){
      // Get the number value at a specific index in a string (0 or 1)
      return +s.charAt(i);
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //original source
    //https://github.com/keesiemeijer/maze-generator
    class Maze1{
      /**
       * @param {number} rows
       * @param {number} cols
       * @param {string} entryType "D"|"H"|"V"
      */
      constructor(rows,cols,entryType="D", options={}){
        let {bias,removeWalls,maxWallsRemove} = options;
        this.bias= bias || "";
        this.removeWalls= removeWalls || 0;
        // Maximum 300 walls can be removed
        this.maxWallsRemove= maxWallsRemove || 300;
        this.ROWS=rows;
        this.COLS=cols;
        this.matrix = [];
        this.entryNodes = this.getEntryNodes(entryType);
      }
      generate(){
        this.getMatrix(this.parseMaze(
          //[visited, nswe]
          _.fill(this.COLS*this.ROWS, ()=> "01111")
        ));
        this.removeMazeWalls();
      }
      toAscii(){
        return ""
      }
      getIO(){
        let X = this.entryNodes.end.gate,
          E= this.entryNodes.start.gate;
        return {start: [E.x, E.y], end: [X.x, X.y]};
      }
      toGrid(){
        let r,out=[];
        this.matrix.forEach((row,y)=> {
          r=[];
          for(let i=0;i<row.length;++i){
            r[i]= row.charAt(i)=="1"?1:0
          }
          out.push(r);
        });
        return out;
      }
      parseMaze(nodes){
        const Position = { n: 1, s: 2, w: 3, e: 4, };
        const Opposite= { n: 2, s: 1, w: 4, e: 3 };
        let max = 0,
          visited = 0,
          moveNodes = [],
          biasCount = 0,
          biasFactor = 3,
          last=nodes.length-1,
          pos= _.randInt(nodes.length);
        if(this.bias){
          if("H" == this.bias){
            biasFactor = (this.COLS/100)>=1 ? _M.ndiv(this.COLS,100) + 2 : 3
          }else if("V" == this.bias){
            biasFactor = (this.ROWS/100)>=1 ? _M.ndiv(this.ROWS,100) + 2 : 3
          }
        }
        // Set start node visited.
        nodes[pos] = replaceAt(nodes[pos], 0, 1);
        let next,dir,dirs;
        while(visited < last){
          ++biasCount;
          ++max;
          next = this.getNeighbours(pos);
          dirs = Object.keys(next).filter(k=>
            -1 != next[k] && !stringVal(nodes[next[k]], 0));
          if(this.bias && (biasCount != biasFactor)){
            dirs = this.biasDirections(dirs);
          }else{
            biasCount = 0;
          }
          if(dirs.length){
            ++visited;
            if(dirs.length>1)
              moveNodes.push(pos);
            dir= dirs[_.randInt(dirs.length)];
            // Update current position
            nodes[pos] = replaceAt(nodes[pos], Position[dir], 0);
            // Set new position
            pos = next[dir];
            // Update next position
            nodes[pos] = replaceAt(nodes[pos], Opposite[dir], 0);
            nodes[pos] = replaceAt(nodes[pos], 0, 1);
          }else{
            if(moveNodes.length==0){ break }
            pos= moveNodes.pop();
          }
        }
        return nodes;
      }
      getMatrix(nodes){
        // Add the complete maze in a matrix where 1 is a wall and 0 is a corridor.
        let row1="",
          row2="",
          N = this.COLS* this.ROWS;
        _.assert(nodes.length == N,"invalid nodes");
        for(let i=0; i<N; ++i){
          row1 += row1.length==0 ? "1" : "";
          row2 += row2.length==0 ? "1" : "";
          if(stringVal(nodes[i], 1)){
            row1 += "11";
            row2 += stringVal(nodes[i], 4)? "01" : "00";
          }else{
            let hasAbove = nodes.hasOwnProperty(i-this.COLS);
            let above = hasAbove && stringVal(nodes[i-this.COLS], 4);
            let hasNext = nodes.hasOwnProperty(i+1);
            let next = hasNext && stringVal(nodes[i+1], 1);
            if(stringVal(nodes[i], 4)){
              row1 += "01";
              row2 += "01";
            }else if(next || above){
              row1 += "01";
              row2 += "00";
            }else{
              row1 += "00";
              row2 += "00";
            }
          }
          if((i+1) % this.COLS == 0){
            this.matrix.push(row1,row2);
            row1 = "";
            row2 = "";
          }
        }
        // Add closing row
        this.matrix.push("1".repeat((this.COLS * 2) + 1));
      }
      getEntryNodes(access){
        let entryNodes = {},
          y = ((this.ROWS * 2) + 1) - 2,
          x = ((this.COLS * 2) + 1) - 2;
        if("D" == access){
          entryNodes.start = { x: 1, y: 1, gate: { x: 0, y: 1 } };
          entryNodes.end = { x, y, gate: { x: x + 1, y } };
        }
        if("H" == access || "V" == access){
          let xy = ("H" == access) ? y : x;
          xy = ((xy - 1) / 2);
          let even = xy % 2 == 0;
          xy = even ? xy + 1 : xy;
          let start_x = ("H" == access) ? 1 : xy;
          let start_y = ("H" == access) ? xy : 1;
          let end_x = ("H" == access) ? x : (even ? start_x : start_x + 2);
          let end_y = ("H" == access) ? (even ? start_y : start_y + 2) : y;
          let startgate = ("H" == access) ? { x: 0, y: start_y } : { x: start_x, y: 0 };
          let endgate = ("H" == access) ? { x: x + 1, y: end_y } : { x: end_x, y: y + 1 };
          entryNodes.start = { x: start_x, y: start_y, gate: startgate };
          entryNodes.end = { x: end_x, y: end_y, gate: endgate };
        }
        return entryNodes;
      }
      biasDirections(dirs){
        let hz = dirs.indexOf("w") != -1 || dirs.indexOf("e") != -1,
          vt = dirs.indexOf("n") != -1 || dirs.indexOf("s") != -1;
        return ("H" == this.bias && hz)?
          dirs.filter(k=> "w" == k || "e" == k)
          : ("V" == this.bias && vt)?
          dirs.filter(k=> "n" == k || "s" == k) : dirs;
      }
      getNeighbours(pos){
        return {
          w: (pos>0 && (pos % this.COLS) != 0) ? pos-1 : -1,
          e: ((pos+1) % this.COLS) != 0 ? pos+1 : -1,
          n: (pos - this.COLS)>=0 ? pos - this.COLS : -1,
          s: ((this.COLS* this.ROWS) > (pos + this.COLS)) ? pos + this.COLS : -1
        }
      }
      removeWall(row, index){
        // Remove wall if possible.
        const evenRow = row % 2 == 0,
          evenIndex = index % 2 == 0,
          wall = stringVal(this.matrix[row], index);
        if(!wall){ return false }
        if(!evenRow && evenIndex){
          // Uneven row and even column
          const hasTop = (row-2 > 0) && (1 == stringVal(this.matrix[row-2], index));
          const hasBottom = (row + 2 < this.matrix.length) && (1 == stringVal(this.matrix[row + 2], index));
          if(hasTop && hasBottom){
            this.matrix[row] = replaceAt(this.matrix[row], index, "0");
            return true;
          }
          if(!hasTop && hasBottom){
            const left = 1 == stringVal(this.matrix[row - 1], index - 1);
            const right = 1 == stringVal(this.matrix[row - 1], index + 1);
            if (left || right) {
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }else if(!hasBottom && hasTop){
            const left = 1 == stringVal(this.matrix[row + 1], index - 1);
            const right = 1 == stringVal(this.matrix[row + 1], index + 1);
            if (left || right) {
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }
        }else if(evenRow && !evenIndex){
          // Even row and uneven column
          const hasLeft = 1 == stringVal(this.matrix[row], index - 2);
          const hasRight = 1 == stringVal(this.matrix[row], index + 2);
          if(hasLeft && hasRight){
            this.matrix[row] = replaceAt(this.matrix[row], index, "0");
            return true;
          }
          if(!hasLeft && hasRight){
            const top = 1 == stringVal(this.matrix[row - 1], index - 1);
            const bottom = 1 == stringVal(this.matrix[row + 1], index - 1);
            if(top || bottom){
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }else if(!hasRight && hasLeft){
            const top = 1 == stringVal(this.matrix[row - 1], index + 1);
            const bottom = 1 == stringVal(this.matrix[row + 1], index + 1);
            if(top || bottom){
              this.matrix[row] = replaceAt(this.matrix[row], index, "0");
              return true;
            }
          }
        }
      }
      removeMazeWalls(){
        if(this.removeWalls==0 ||
           this.matrix.length==0){ return }
        let min = 1,
          tries = 0,
          max = this.matrix.length - 1,
          row,walls, maxTries = this.maxWallsRemove;
        while(tries < maxTries){
          ++tries;
          // Did we reached the goal
          if(this.wallsRemoved >= this.removeWalls){ break }
          // Get random row from matrix
          let y =_.randInt2(min,max);
          y = (y == max) ? y - 1 : y;
          walls = [];
          row = this.matrix[y];
          // Get walls from random row
          for(let w,i = 0; i < row.length; i++){
            if(i == 0 || i == row.length - 1){ continue }
            if( stringVal(row, i)) walls.push(i);
          }
          // Shuffle walls randomly
          _.shuffle(walls);
          // Try breaking a wall for this row.
          for(let i = 0; i < walls.length; i++){
            if(this.removeWall(y, walls[i])){
              // Wall can be broken
              ++this.wallsRemoved;
              break;
            }
          }
        }
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    //original
    //https://weblog.jamisbuck.org/2010/12/27/maze-generation-recursive-backtracking
    class Maze2{
      static DV={n:1,s:2,e:4,w:8};
      static DX = { e: 1, w: -1, n: 0, s: 0 };
      static DY = { e:0, w: 0, n: -1, s: 1 };
      static OPPOSITE= { e: "w", w:"e", n:"s",s:"n" };

      /**
       * @param {number} width
       * @param {number} height
       */
      constructor(width,height){
        this.COLS=width;
        this.ROWS=height;
        this._getEntryNodes();
      }
      _getEntryNodes(){
        let nodes = {},
          y = ((this.ROWS * 2) + 1) - 2,
          x = ((this.COLS * 2) + 1) - 2;
        nodes.start = { x: 1, y: 1, gate: { x: 0, y: 1 } };
        nodes.end = { x, y, gate: { x: x + 1, y } };
        this.entryNodes=nodes;
      }
      getIO(){
        let X = this.entryNodes.end.gate,
          E= this.entryNodes.start.gate;
        return {start: [E.x, E.y], end: [X.x, X.y]};
      }
      generate(){
        this.grid=this._walk(0, 0, _.fill(this.ROWS, ()=> _.fill(this.COLS, 0)))
      }
      canSouth(v){
        return (v & Maze2.DV["s"]) != 0
      }
      canEast(v){
        return (v & Maze2.DV["e"]) != 0
      }
      toAscii(){
        let g=this.grid,
          height= g.length,
          out=[], width= g[0].length;
        for(let s, y=0;y<height;++y){
          s="|";
          for(let x=0;x<width;++x){
            s+= this.canSouth(g[y][x]) ? " " : "_";
            if(this.canEast(g[y][x])){
              s+= this.canSouth(g[y][x] | g[y][x+1]) ? " " : "_";
            }else{
              s+= "|";
            }
          }
          out.push(s);
        }
        out.unshift("_".repeat(out[0].length));
        return out.join("\n");
      }
      _walk(cx, cy, grid){
        let nx,ny,dirs = _.shuffle(["n","s","e","w"]);
        dirs.forEach(d=>{
          nx = cx + Maze2.DX[d];
          ny = cy + Maze2.DY[d];

          if(ny>=0&&ny<grid.length &&
             nx>=0&&nx<grid[ny].length && grid[ny][nx]==0){
            grid[cy][cx] |= Maze2.DV[d];
            grid[ny][nx] |= Maze2.DV[Maze2.OPPOSITE[d]];
            this._walk(nx, ny, grid);
          }
        });
        return grid;
      }
      toGrid(){
        let g=this.grid,
          height=g.length,
          r1,r2,out=[], width=g[0].length;
        for(let k,s, y=0;y<height;++y){
          r1=[];
          r2=[];
          k=0;
          r1[k]=1;
          r2[k]=1;
          for(let x=0;x<width;++x){
            ++k;
            if(this.canSouth(g[y][x])){
              r1[k]=0;
              r2[k]=0;
            }else{
              r1[k]=0;
              r2[k]=1;
            }
            ++k;
            if(this.canEast(g[y][x])){
              r1[k]=0;
              r2[k]=1;
            }else{
              r1[k]=1;
              r2[k]=1;
            }
          }
          out.push(r1,r2);
        }
        out.unshift(_.fill(out[0].length, 1));
        return out;
      }
    }

    //;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    const _$={
      Maze1, Maze2
    };

    return _$;
  }

  //export--------------------------------------------------------------------
  if(typeof module == "object" && module.exports){
    module.exports=_module(require("../main/core"),
                           require("../main/math"));
  }else{
    gscope["io/czlab/mcfud/algo/maze"]=_module
  }

})(this);


