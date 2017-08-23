class Layer(val dim:Int, val index:Int){
	val neurons:Array[Neuron] = buildNeurons()
	
	//建立Layer底下的Neuron與Weights
	def buildNeurons():Array[Neuron] = {
	  var temp:Array[Neuron] = new Array[Neuron](dim)
	  //2014/6/22 新增平行
	  for (i <- (0 until dim).par){
	    temp(i) = new Neuron(index,i)
	  }
	  temp
	}
	
	//計算激發狀態值
	def activate(input:Array[Double]){
	  if (index==0){
	    for (i <- 0 until neurons.length){
	      neurons(i).stateResult = input(i)
	    }
	  }
	  else neurons.par.foreach((p:Neuron)=>p.calculateStateResult(input))
	}
	
	//把整個Layer的Neuron之激發狀態值的組合做為更上一個Layer的input
	def getLayerOutput():Array[Double] = {
	  //printLayerOutput()
	  Array.tabulate(neurons.length)(i => neurons(i).stateResult)
	}
	
	def printLayerOutput(){
	  for (i <- 0 until dim){
	    print(neurons(i).stateResult + "\t")
	  }
	  println()
	}
	
	def propogate(target:Array[Double]){
	  neurons.par.foreach((p:Neuron)=>p.calculateError(target))
	}
	
	def update(){
	   neurons.par.foreach((p:Neuron)=>p.updateWeight_Threshold)
	}
	
	def printWeight_Threshold(){
	  for (i <- 0 until dim){
	    neurons(i).printWeight_Threshold()
	    var waitPoint = readLine("請輸入任意鍵繼續")
	  }
	  println()
	}
	
	def printError(){
	  for (i <- 0 until dim){
	    neurons(i).printError()
	  }
	  println()
	}
	
	def refresh(){
	  neurons.par.foreach((p:Neuron)=>p.refresh)
	}
}