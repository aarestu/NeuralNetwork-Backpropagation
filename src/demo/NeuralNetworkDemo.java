/*
 * Copyright 2014 aarestu.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package demo;

import net.Net;
import net.TransferFunction;

/**
 *
 * @author aarestu
 */
public class NeuralNetworkDemo {
    
    public static void main(String[] args) {
        
        Net net = new Net();
               
        int[] topology = new int[3];
        topology[0] = 2;
        topology[1] = 4;
        topology[2] = 1;
        
        net.setTopology(topology);

        double[][] input = new double[4][2];
        double[][] target = new double[4][1];

        input[0][0]     = 0.0;
        input[0][1]     = 0.0;
        target[0][0]    = -1.0;

        input[1][0]     = 0.0;
        input[1][1]     = 1.0;
        target[1][0]    = 1.0;

        input[2][0]     = 1.0;
        input[2][1]     = 0.0;
        target[2][0]    = 1.0;

        input[3][0]     = 1.0;
        input[3][1]     = 1.0;
        target[3][0]    = -1.0;

        //update parameter
        TransferFunction.TRANSFER_FUNCTION = TransferFunction.TRANSFER_FUNCTION_TANH;
        Net.learningrate    = 0.25;
        Net.momentum        = 0.5;
        
        double error = 0.0;
        int epoc = 0;
        do {
            net.sum_global_error = 0.0;

            for (int i = 0; i < 4; i++) {
                net.feedForward(input[i]);
                net.backProp(target[i]);
            }
            error = Math.sqrt(net.sum_global_error / (4 * topology[topology.length - 1]));
            epoc++;
            if(epoc % 100 == 0){
                System.out.println("learning masih proses di epoc " + epoc + ". dengan error " + error);
            }

        } while ( error > 0.01 );
        
        System.out.println("Learning berhenti di epoc " + epoc + ". dengan error " + error);
        
        System.out.println("\nCoba Respon NN :");
        double[] hasil;
        for (int i = 0; i < 4; i++) {
            net.feedForward(input[i]);
            hasil = net.getHasil();
            System.out.println( input[i][0] + " xor "+input[i][1]+" = " + hasil[0]);
        }
    }
}
