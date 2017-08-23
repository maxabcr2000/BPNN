object Controller {
	var inputDim = 0
	var hiddenLayerCount = 1
	var hiddenDims:Array[Int] = Array()
	var outputDim = 0
	var wUpperBound = 0.0
	var wLowerBound = 0.0
	var transformMethodChoice:Int = 2
	
	var learningData:Vector[Array[Double]] = Vector[Array[Double]]()
    var desiredTargets:Vector[Array[Double]] = Vector[Array[Double]]()
	var learnGain = 0.0
    var executingData:Vector[Array[Double]] = Vector[Array[Double]]()
    var learnDataIndex:Int = 0
    var learnGeneration:Int = 0
    
	def main(args: Array[String]) {
		inputDim = readLine("請輸入Input層的維度：").toInt
		//hiddenLayerCount = readLine("請輸入Hidden層的層數：").toInt
		hiddenDims = new Array[Int](hiddenLayerCount)
		for (i <- 0 until hiddenLayerCount){
		  hiddenDims(i) = readLine("請輸入Hidden層第" + (i+1) + "層的維度：").toInt
		}
		outputDim = readLine("請輸入Output層的維度：").toInt
        wUpperBound = readLine("請輸入Neuron Weight的上界值：").toDouble
        wLowerBound = readLine("請輸入Neuron Weight的下界值：").toDouble
		//transformMethodChoice = readLine("請選擇轉換函數\n1.雙彎曲(Sigmoid)函數\n2. 雙曲正切(Hyperbolic-tangent)函數\n").toInt
		learnGain = readLine("請輸入學習速率值：").toDouble
		
		//讀取學習資料
		for(line <- scala.io.Source.fromFile("Learn_Data.txt").getLines()){
	      //for testing
		  //println("readline:" + line)
		  if (line.length>2)
		  {
			  val lineSplit = line.split("/")
			  val learnSplit = lineSplit(0).split(" ")
			  val targetSplit = lineSplit(1).split(" ")
			  
			  //println("lineSplit:" + lineSplit.length)
			  //println("learnSplit:" + learnSplit.length)
			  //println("targetSplit:" + targetSplit.length)
			 
	    	  learningData = learningData:+learnSplit.map(_.toDouble)  
	    	  desiredTargets = desiredTargets:+targetSplit.map(_.toDouble)  
	    	  //var waitPoint = readLine("輸入隨意值繼續")
		  }
		}   
		
		//讀取執行資料
		for(line <- scala.io.Source.fromFile("Execute_Data.txt").getLines()){
	      val lineSplit = line.split(" ")
    	  executingData = executingData:+lineSplit.map(_.toDouble)  
		} 
		
		//建立類神經網路
		NeuronNetwork.init()
		
		//測試用，觀察類神經網路最初的狀態
		//NeuronNetwork.testWeights_Threshold()
		
		//學習階段
		learnGeneration = 0
		var preTotalError = -100.0
		do
		{
		    val startTime = System.currentTimeMillis()
		    NeuronNetwork.refresh()
		    learnGeneration = learnGeneration + 1
			println("-------------------------Learning "+ learnGeneration +"-------------------------")
		    //c屬於k
		    //2014/6/22 新增平行
		    for(i <- (0 until learningData.length).par){
	          learnDataIndex = i
	          //println("<InputLayer>")
	          NeuronNetwork.learn(learningData(i),desiredTargets(i))	          
	        }
		    println("E(Z):" + NeuronNetwork.totalError)
		    modifyLearnGain(NeuronNetwork.totalError, preTotalError)
		    if (NeuronNetwork.totalError>Math.pow(10.0,-3)){
		      //println("Update!!!")
		      NeuronNetwork.updateZ()
		    }
		    
			//測試用，觀察學習之後類神經網路的狀態
			//NeuronNetwork.testWeights_Threshold()
			
		    //測試用，觀察每個Neruon的Error值
		    //NeuronNetwork.testError()
		    
		    val endTime = System.currentTimeMillis()
		    preTotalError = NeuronNetwork.totalError
		    println("Time Cost: " + (endTime-startTime))
		}while(NeuronNetwork.totalError>Math.pow(10.0,-3))
		

		  
		//執行階段
		println("-------------------------Executing-------------------------")
//        executingData.foreach((data:Array[Double])=> {
//          //println("<InputLayer>")
//          NeuronNetwork.execute(data)
//        })
	}
	
	def modifyLearnGain(curError:Double, preError:Double){
	  if (preError>0)
	  {
		  if (curError<preError) {
		    learnGain = learnGain*1.1
		    println("LearnGain become greater!!!" + learnGain)
		    }
		  else if (curError>preError){
		    learnGain = learnGain * 0.8
		    println("LearnGain become smaller!!!" + learnGain)
		    }
	  }
	}
	
}