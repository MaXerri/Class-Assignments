import java.awt.*;
import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;

class Main {
    public static void main(String[] args) throws IOException {
        //BufferedReader br = new BufferedReader(new FileReader("test.txt"));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        int n = -1;
        ArrayList<Integer> radii = new ArrayList<>();
        Integer[] radius_array = new Integer[0];
        ArrayList<ArrayList<Integer>> opt = new ArrayList<>();

        // process input data
        String line;
        while ((line = br.readLine())!= null) {
            if (n == -1) {
                n = Integer.valueOf(line);
            }
            else{
                String[] radii_splitted = line.split(" ");
                for (int i=0; i< radii_splitted.length;i++){
                    if (i==0){
                        radii.add(1);
                    }
                    radii.add(Integer.valueOf(radii_splitted[i]));
                }
                radii.add(1);

            }
        }

        // fill 2D array with zeros boi

        for (int l = 0; l<radii.size();l++){
            ArrayList<Integer> i = new ArrayList<>();
            for (int j =0;j<radii.size();j++){
                i.add(0);
            }
            opt.add(i);
            //System.out.println(radii.get(l));
        }

        int answer = dp_algo(radii,opt,0,radii.size()-1);

        //output

        bw.write(String.valueOf(answer));
        bw.write('\n');

        bw.flush();
        br.close();
        bw.close();
    }

    public static int dp_algo (ArrayList<Integer> a,ArrayList<ArrayList<Integer>> opt,int l,int r){

        // algo implementation
        int maax = 0;
        int max_m = -1;
        if (r-l<=1){
            //System.out.println("bc");
            return (0);
        }
        else if (opt.get(l).get(r)!=0) {
            return (opt.get(l).get(r));
        }
        else{
            int iteration_max = 0;
            for (int m = 1; m<r-l;m++){

                iteration_max = dp_algo(a,opt,l,m+l) + dp_algo(a,opt,m+l,r) + a.get(l+m)*a.get(l)*a.get(r);
                //System.out.println(iteration_max +" "+ String.valueOf(m));
                if (iteration_max > maax){
                    maax = iteration_max;
                    max_m = m;
                }
            }
            opt.get(l).set(r,maax);
            return maax;
        }
    }
}