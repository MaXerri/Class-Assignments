import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.Arrays;
class Main {
    public static void main(String[] args) throws IOException {
        //BufferedReader br = new BufferedReader(new FileReader("test2.txt"));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        int n_read = 0;
        int[] r_lst = {};
        int[] c_lst;
        int[][] g={{0,0}};

        int n = 0;
        int m = 0;
        int delta=0;

        // process input data
        String line;
        while ((line = br.readLine())!= null) {
            if (n_read == 0) {
                n = Integer.valueOf(line.split(" ")[0]);
                m = Integer.valueOf(line.split(" ")[1]);

            }
            else if (n_read==1){
                r_lst = new int[n];
                String[] r_lst_string = (line.split(" "));
                for (int i = 0;i <r_lst_string.length;i++){
                    r_lst[i] = Integer.valueOf(r_lst_string[i]);
                }

            }
            else{
                if (n_read == 2){
                    g = new int[n][n];
                }
                c_lst = new int[m];
                String[] spl = line.split(" ");
                c_lst[n_read-2] = Integer.valueOf(spl[2]);
                g[Integer.valueOf(spl[0])-1][Integer.valueOf(spl[1])-1] = Integer.valueOf(spl[2]);
            }
            n_read++;
        }

        //calculate delta

        for (int i = 0;i<g.length;i++){
            if (r_lst[i]>0){
                delta=delta+r_lst[i];
            }
        }

        /**
        for (int i =0;i<n;i++){
            for (int j =0;j<n;j++){
                System.out.print(g[i][j] + " ");
            }
            System.out.println("");
        }
        */

        //run ff on G

        int[][] f = ff(g);

        /**
        System.out.println("FF output");
        for (int i =0;i<f[0].length;i++){
            for (int j =0;j<f[0].length;j++){
                System.out.print(f[i][j] + " ");
            }
            System.out.println("");
        }
         */


        int[][] g_r = new int[g.length][g.length];

        //creating residual graph
        for (int i = 0; i<g.length;i ++){
            for (int j = 0; j<g.length;j ++){
                if (f[i][j] < g[i][j]){
                    g_r[i][j] = g[i][j] - f[i][j];
                }
                if (f[i][j] > 0){
                    g_r[j][i] = f[i][j];
                }
            }
        }

        /**
        System.out.println("");
        System.out.println("Gf output");

         for (int i =0;i<f[0].length;i++){
            for (int j =0;j<f[0].length;j++){
                System.out.print(g_r[i][j] + " ");
            }
            System.out.println("");
        }
        */

        //make deep copy of g_r
        int[][] gr_mod = new int[g.length][g.length];
        for (int i= 0;i<g.length;i++){
            for (int j= 0;j<g.length;j++){
                gr_mod[i][j] = g_r[i][j];
            }
        }


        // get sets A_s and B_t
        boolean[] as = bfs_As(1,g_r);
        boolean[] bt = bfs_Bt(g.length,g_r);

        //System.out.println(Arrays.toString(as));
        //System.out.println(Arrays.toString(bt));
        //System.out.println(Arrays.toString(r_lst));

        //creating new graph

        for (int i=0;i<g.length;i++){
            for (int j=0;j<g.length;j++){
                if (as[i]==true || as[j]==true || bt[i]==true || bt[j]==true){
                    gr_mod[i][j] = 0;
                    gr_mod[j][i] = 0;
                }
                else{
                    if (i!=j && g_r[i][j]>0){
                        gr_mod[i][j] = delta + 1;
                    }
                }
            }
        }
        for (int i=1;i<g.length-1;i++){
            if (r_lst[i]>0 && as[i]==false && bt[i]==false){
                gr_mod[0][i] = r_lst[i];
            }
            if (r_lst[i]<0 && as[i]==false && bt[i]==false){
                gr_mod[i][g.length-1] = -r_lst[i];
            }
        }
        //System.out.println("delta: " + delta);
        //printing input to project selection
        /**
        System.out.println("printing input to psec");
        for (int i= 0;i<g.length;i++){
            for (int j= 0;j<g.length;j++){
                System.out.print(gr_mod[i][j]);
            }
            System.out.println("");
        }
        */

        //run ff again t to find max flow, thus min cut
        int[][] psec = ff(gr_mod);
        //boolean[] min_cut= bfs_As(1,gr_mod);

        //printing proj selection output

        /**
        System.out.println("Printing flow from project selection");
        for (int i= 0;i<g.length;i++){
            for (int j= 0;j<g.length;j++){
                System.out.print(psec[i][j]);
            }
            System.out.println("");
        }
        */

        int[][] ff_res = new int[g.length][g.length];
        //creating residual graph
        for (int i = 0; i<g.length;i ++){
            for (int j = 0; j<g.length;j ++){
                if (psec[i][j]  < gr_mod[i][j]){
                    ff_res[i][j] = gr_mod[i][j] - psec[i][j];
                }
                if (psec[i][j] > 0){
                    ff_res[j][i] = psec[i][j];
                }
            }
        }

        //printing res final
        /**
        System.out.println("printing residual of psec");
        for (int i= 0;i<g.length;i++){
            for (int j= 0;j<g.length;j++){
                System.out.print(ff_res[i][j]);
            }
            System.out.println("");
        }
        */

        boolean[] reachable = bfs_As(1,ff_res);
        //System.out.println(Arrays.toString(reachable));
        //summing A'
        int sum = 0;
        for (int i=1;i<g.length-1;i++){
            if (reachable[i]==true && as[i]==false && bt[i]==false){
                sum+=r_lst[i];
            }
        }
        //System.out.println(sum + " A'");

        // summing profits from As
        for (int i =0;i<as.length;i++){
            if (as[i]==true){
                sum+=r_lst[i];
            }
        }

        bw.write(String.valueOf(sum));
        bw.write('\n');
        bw.flush();
        br.close();
        bw.close();
    }
    public static boolean[] bfs_As(int s,int[][] res){

        boolean reached[] = new boolean[res.length];

        LinkedList<Integer> q = new LinkedList<>();

        reached[s-1] = true;
        q.add(s);

        while (q.size()!=0){
            int node = q.pollFirst();

            for (int i = 0; i <res[node-1].length;i++){
                if (reached[i]==false && res[node-1][i]>0){

                    reached[i] = true;
                    q.add(i+1);

                }
            }
        }
        return reached;
    }

    public static boolean[] bfs_Bt(int t,int[][] res){

        boolean reached[] = new boolean[res.length];

        LinkedList<Integer> q = new LinkedList<>();

        reached[t-1] = true;
        q.add(t);

        while (q.size()!=0){
            int node = q.pollFirst();

            for (int i = 0; i <res[node-1].length;i++){
                if (reached[i]==false && res[i][node-1]>0){
                    reached[i] = true;
                    q.add(i+1);
                }
            }
        }
        return reached;
    }


    public static int[] bfs(int s, int t,int[][] res,int[][] g){
        boolean reached[] = new boolean[res.length];

        LinkedList<Integer> q = new LinkedList<>();
        int[] backtrack_lst = new int[res.length];
        reached[s-1] = true;
        q.add(s);

        while (q.size()!=0){
            int node = q.pollFirst();

            for (int i = 0; i <res[node-1].length;i++){
                if (reached[i]==false && res[node-1][i]>0){
                    if (i+1==t){
                        backtrack_lst[i] = node;
                        return backtrack_lst;
                    }
                    reached[i] = true;
                    q.add(i+1);
                    backtrack_lst[i] = node;
                }
            }
        }
        return backtrack_lst;
    }

    public static int[][] ff( int[][] g){

        int[][] res_cap = new int[g.length][g.length];

        //deep clone
        for (int i= 0;i<g.length;i++){
            for (int j= 0;j<g.length;j++){
                res_cap[i][j] = g[i][j];
            }
        }

        int[][] flow = new int[g[0].length][g[0].length];

        while (true) {
            int[] backtrack_lst = bfs(1,g.length,res_cap,g);
            if (backtrack_lst[g.length-1]!=0){

                //augment

                int b = 100000; //min capacity of edges in Path P
                for (int v = g.length; v!=1; v = backtrack_lst[v-1]){
                    int u = backtrack_lst[v-1];
                    if (res_cap[u-1][v-1] < b){ //make sure this never picks up a non-existing edge and thus a zero
                        b = res_cap[u-1][v-1];
                    }
                }
                //create a new flow f' on G as follows
                for (int v = g.length; v!=1; v = backtrack_lst[v-1]){
                    int u = backtrack_lst[v-1];

                    //update capacity array
                    res_cap[u-1][v-1] = res_cap[u-1][v-1] -b;
                    res_cap[v-1][u-1] = res_cap[v-1][u-1] +b;

                    //update flow array
                    if (g[u-1][v-1]>0){
                        flow[u-1][v-1] = flow[u-1][v-1] +b;
                    }
                    else{
                        flow[v-1][u-1] = flow[v-1][u-1] - b;
                    }
                }
            }
            else{
                break;
            }
        }
        return flow;


    }


}