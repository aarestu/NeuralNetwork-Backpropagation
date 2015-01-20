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

package com.aarestu.nn.net;

/**
 *
 * @author aarestu
 */
public class TransferFunction {
    public static int TRANSFER_FUNCTION = 2;
    public static final int TRANSFER_FUNCTION_TANH = 1;
    public static final int TRANSFER_FUNCTION_SIGMOID = 2;
    public static final int TRANSFER_FUNCTION_BIPOLAR = 3;
    public static final int TRANSFER_FUNCTION_IDENTITAS = 4;
    
    public static double transfer(double x){
        switch(TRANSFER_FUNCTION){
            case TRANSFER_FUNCTION_TANH :
                return tanh(x);
            case TRANSFER_FUNCTION_BIPOLAR :
                return bipolar(x);
            case TRANSFER_FUNCTION_IDENTITAS :
                return identitas(x);
            default:
                return sigmoid(x);
        }
    }
    
    public static double transferTurunan(double x){
        switch(TRANSFER_FUNCTION){
            case TRANSFER_FUNCTION_TANH :
                return tanhTurunan(x);
            case TRANSFER_FUNCTION_BIPOLAR :
                return bipolarTurunan(x);
            case TRANSFER_FUNCTION_IDENTITAS :
                return identitas(x);
            default:
                return sigmoidTurunan(x);
        }
    }
    
    private static double tanh(double x) {
        //tanh - output range [-1.0 s/d 1.0]
        return Math.tanh(x);
    }

    private static double tanhTurunan(double x) {
        //turunan sigmoid
        return 1 - x * x;
    }
    
    private static double sigmoid(double x) {
        //sigmoid - output range [0.0 s/d 1.0]
        return 1 / (1 + Math.exp(-x));
    }

    private static double sigmoidTurunan(double x) {
        //turunan sigmoid
        return sigmoid(x) * (1 - sigmoid(x));
    }
    
    private static double bipolar(double x) {
        //bipolar - output range [-1.0 s/d 1.0]
        return ( ( 2 / ( 1 + Math.exp(-x) ) ) - 1 );
    }

    private static double bipolarTurunan(double x) {
        //turunan bipolar
        return ( 1 - bipolar(x) * bipolar(x) ) / 2;
    }
    
    private static double identitas(double x) {
        return x;
    }

}
