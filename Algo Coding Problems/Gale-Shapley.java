import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.StringTokenizer;

class Main {
    public static void main(String[] args) throws IOException{
        //BufferedReader br = new BufferedReader(new FileReader("Test.txt"));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        int m = -1;
        int n = -1;
        int line_num = 0;
        int hospital_count = 1;
        int net_capacity = 0;
        ArrayList<Integer> capacity= new ArrayList<>();
        ArrayList<String[]> hos_pref = new ArrayList<>();
        ArrayList<String[]> res_pref = new ArrayList<>();
        LinkedList<Integer> free_hos = new LinkedList<>();
        ArrayList<Integer> next = new ArrayList<>();
        ArrayList<Integer> current = new ArrayList<>();
        ArrayList<Integer> capacity_counter = new ArrayList<>();
        ArrayList<Integer> zeros = new ArrayList<>();

        //setting up data types

        String line;
        while ((line = br.readLine())!= null){
            if (n==-1 && m==-1 && line_num ==0){
                String[] splitted = line.split(" ");
                m = Integer.parseInt(splitted[0]);
                n = Integer.parseInt(splitted[1]);
            }

            if (line_num>0 && line_num<=m && m!=-1){
                capacity.add(Integer.valueOf(line));
                capacity_counter.add(Integer.valueOf(line));
                zeros.add(0);
                net_capacity+=Integer.valueOf(line);

            }
            if (line_num>m && line_num<=(2*m) && m!=-1){
                hos_pref.add(line.split(" "));
                next.add(1);
                free_hos.add(hospital_count);
                hospital_count++;

            }

            if (line_num>(2*m) && m!=-1){
                String[] array_list = line.split(" ");

                //implementing Ranking Array for Faster Comparisons
                String[] array_list_ranked = new String[array_list.length];
                for (int i = 0; i<array_list.length;i++){
                    array_list_ranked[Integer.parseInt(array_list[i])-1] = String.valueOf(i+1);
                }
                res_pref.add(array_list_ranked);
                current.add(0);
            }
            line_num ++;
        }


        // running the algorithm
        while (free_hos.size()!=0){
            int hospital = free_hos.get(0);
            int res_in_contention = Integer.parseInt((hos_pref.get(hospital-1))[next.get(hospital-1)-1]);

            if (current.get(res_in_contention-1) == 0){
                current.set(res_in_contention-1,hospital);
                capacity_counter.set(hospital-1,capacity_counter.get(hospital-1) - 1);
                if (capacity_counter.get(hospital-1) ==0 ){
                    free_hos.pollFirst();
                }
            }
            else{
                int current_hospital = current.get(res_in_contention-1);
                if (Integer.parseInt(res_pref.get(res_in_contention-1)[hospital-1]) < Integer.parseInt(res_pref.get(res_in_contention-1)[current_hospital-1]) ){
                    current.set(res_in_contention-1,hospital);
                    capacity_counter.set(hospital-1,capacity_counter.get(hospital-1)-1);
                    if (capacity_counter.get(hospital-1) ==0 ){
                        int ded = free_hos.pollFirst();
                    }

                    if (capacity_counter.get(current_hospital-1) ==0 ){
                        free_hos.addFirst(current_hospital);
                    }
                    capacity_counter.set(current_hospital -1,capacity_counter.get(current_hospital-1)+1);
                }
            }
            next.set(hospital-1,next.get(hospital-1) +1);
        }



        //convert back to referencing hospitals
        /**
        ArrayList<Integer> current_out = new ArrayList<>();

        for (int i =0; i<current.size();i++){
            int counter = 0;
            int counter_total=0;
            while (current.get(i)>counter_total){
                counter_total+= capacity.get(counter);
                counter++;
            }
            current_out.add(counter);
        }
        */

        // output to stdout
        for (int i = 0; i< n;i++){
            //System.out.println(current_out.get(i));
            bw.write(String.valueOf(current.get(i)));
            bw.write('\n');

        }
        bw.flush();
        br.close();
        bw.close();
    }
}