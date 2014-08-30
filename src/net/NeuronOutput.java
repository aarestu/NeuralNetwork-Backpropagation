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

package net;

/**
 *
 * @author aarestu
 */
public class NeuronOutput extends Neuron {
    
    private double target;
    
    public NeuronOutput(int myIndex) {
        super(myIndex);
    }
    
    public void hitungGalat(){
        galat = getOutputError() * TransferFunction.transferTurunan(this.outputVal);
    }
    
    public void setTarget(double target){
        this.target = target;
    }
    
    public double getOutputError(){
        return  this.target - this.getOutputVal();
    }
    
    
}
