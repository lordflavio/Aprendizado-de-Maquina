package MLP;

public class MLP_GRADIENTE {
	
	double[][] input; /* base de treino */
	double[] output; /* saida da bade de traino */
	
	double[][] inputValidate; /* base de validação */
	double[] outputValidate; /* saida da base validação */
	
	int hiddenNeurons; /* Quantidade de neuronios escondidos */
	double learning; /* taxa de aprendisado  */
	
	double[][] inputWeights; /* pesos do treino */
	double[][] outputWeights; /* peso da saida do treino */
	
	double[] biasInputWeights; /* peso do bias */
	double[] biasOuputWeights; /* peso saida bias */
	
	double[] erroValidate; /* Erro  de Validação */
	
	
	/* Metodo Construtor */
	public MLP_GRADIENTE(double[][] input, double[] output, double[][] inputValidate,double[] outputValidate, int hiddenNeurons, double learning ) {
		
		this.input = input;
		this.output = output;
		this.inputValidate = inputValidate;
		this.outputValidate = outputValidate;

		this.hiddenNeurons = hiddenNeurons;
		this.learning = learning;

		inputWeights = generateWeights(this.input[0].length, this.hiddenNeurons);
		outputWeights = generateWeights(this.hiddenNeurons, 1);
		
		biasInputWeights = generateBiasWeights(this.hiddenNeurons);
		biasOuputWeights = generateBiasWeights(1);
		
		//inputWeights = new double[this.input[0].length][this.hiddenNeurons];
		//outputWeights = new double [this.hiddenNeurons][1];
		
		//biasInputWeights = new double[hiddenNeurons];
		//biasOuputWeights = new double[1];
		
//		this.generateMlp(inputTeste, outputTeste);
		
	}
	
	/* Metodo de Treino */
	public double[] train (int epoca) {
		
		this.erroValidate = new double[epoca];
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		
		double erro = 0;
		
		double[] erroTotal = new double[epoca];  
		
		double[] gradients = new double[this.hiddenNeurons]; 
		double gradientOut = 0;
		
		for (int n = 0; n < epoca; n++) { /* Número de interações */
			
			for (int i = 0; i < input.length; i++) { /* Linhas da Base  */
				
				for (int j = 0; j < this.hiddenNeurons; j++) { /* Neuronios escondiodos */
					
					for (int k = 0; k < input[0].length; k++) { /* coulunas da base*/
						
						net[j] +=  this.inputWeights[k][j] * this.input[i][k];
					}
					
						net[j] += this.biasInputWeights[j];
															
						net[j] = sigmoid(net[j]);
				}
				
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					 
						netOut += this.outputWeights[j][0] * net[j];
					   
				}
				
				netOut += this.biasOuputWeights[0];
				
				netOut = sigmoid(netOut);
				
			//	System.out.println("Saida desejada: =>"+this.output[i] +" | Saida Obtida =>" + netOut);
				
				erro = (this.output[i] - netOut);
				
				erroTotal[n] += Math.pow(erro, 2);
				
				
				gradientOut = erro * netOut * (1 - netOut); /*calculo do gradiente do neurônio de saida */
				
				for (int j = 0; j < gradients.length; j++) { /*calculo do gradiente dos neurônio escondidos */
					gradients[j] = this.outputWeights[j][0] * gradientOut;
					gradients[j] = gradients[j] * net[j] * (1 - net[j]);
				}
				
				/* Ajuste de pesos */
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					
					this.outputWeights[j][0] += this.learning * net[j] * gradientOut;
					
					for (int g = 0; g < this.input[0].length; g++) {
						this.inputWeights[g][j] += this.learning * this.input[i][g] * gradients[j]; 
					}
				}
				
				/* Ajuste de Bias */
				for (int j = 0; j < this.biasOuputWeights.length; j++) {
					this.biasOuputWeights[j] += this.learning * 1 * gradientOut;
				}
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					this.biasInputWeights[j] += this.learning * 1 * gradients[j];
					
				}
				
				/* Set de Variaveis */
				
				netOut = 0;
				gradientOut = 0;
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					
					net[j] = 0;
					gradients[j] = 0;
				}

				//System.out.println("------------------------------------------------------------");

			}
			
			
			this.erroValidate[n] = this.validate(this.inputValidate, this.outputValidate);
			
			erroTotal[n] = erroTotal[n] / this.input.length;
			
		}
		
		return erroTotal;
	}
	
	/* Metodo de Validação */
	public double validate(double[][] inputValidate, double[] outputValidate) {
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		double erro = 0;
        double erroTotal = 0;  
		double[] gradients = new double[this.hiddenNeurons]; 
		double gradientOut = 0;
		
		for (int i = 0; i < inputValidate.length; i++) {
			for (int j = 0; j < this.hiddenNeurons; j++) {
				for (int k = 0; k < inputValidate[0].length; k++) {
					net[j] +=  this.inputWeights[k][j] * inputValidate[i][k];
				}
				
				net[j] += this.biasInputWeights[j];
				
				net[j] = sigmoid(net[j]);
			}
			
			for (int j = 0; j < this.hiddenNeurons; j++) {  
				netOut += this.outputWeights[j][0] * net[j];
			}
			
			netOut += this.biasOuputWeights[0];	 
			
			netOut = sigmoid(netOut);
			
			erro = (outputValidate[i] - netOut);
			
			erroTotal += Math.pow(erro, 2);
			
			netOut = 0;
			
			for (int j = 0; j < this.hiddenNeurons; j++) {
				net[j] = 0;
			}
			
		}
		
		return (erroTotal/inputValidate.length);
	}
	
	/* Metodo para faze de testes e previsão da rede */
	
	public double[] generateMlp (double[][] inputTest, double[] outputTest){
		
		double[] result = new double[inputTest.length];
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		double erro = 0;
        double erroTotal = 0;  
		double[] gradients = new double[this.hiddenNeurons]; 
		double gradientOut = 0;
		
		for (int i = 0; i < inputTest.length; i++) {
			for (int j = 0; j < this.hiddenNeurons; j++) {
				for (int k = 0; k < inputTest[0].length; k++) {
					net[j] +=  this.inputWeights[k][j] * inputTest[i][k];
				}
				
				net[j] += this.biasInputWeights[j];
				
				net[j] = sigmoid(net[j]);
			}
			
			for (int j = 0; j < this.hiddenNeurons; j++) {  
				netOut += this.outputWeights[j][0] * net[j];
			}
			
			netOut += this.biasOuputWeights[0];	 
			
			netOut = sigmoid(netOut);
			
			result[i] = netOut;
			
			erro = (outputTest[i] - netOut);
			
			erroTotal += Math.pow(erro, 2);
			
			netOut = 0;
			
			for (int j = 0; j < this.hiddenNeurons; j++) {
				net[j] = 0;
			}
			
		}
		
		return result;
	}
	
	/* Gegar pesos aleatorios para arrays[][] */
	public double[][] generateWeights(int line, int column) {
		double[][] weights = new double[line][column];
		
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[0].length; j++) {
					weights[i][j] = Math.random();
				}
			}
		
		return weights;
	}
	
	/* Gerar pesos alearorios para vetores[] */
	public double[] generateBiasWeights(int line) {
		double[] weights = new double[line];
		
			for (int i = 0; i < weights.length; i++) {
					weights[i] = Math.random();
			}
		
		return weights;
	}
	
	public double sigmoid (double value) {
		return 1/( 1 + Math.exp(-value));
	}

	public double[] getErroValidate() {
		return erroValidate;
	}
}
