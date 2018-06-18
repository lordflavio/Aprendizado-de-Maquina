package MLP;

public class Bases {
	/* Base cota��o do bitcoin em d�lar desde  2017-01-01 � 2018-06-09*/
	private double[] base = {997.6888,1018.05,1030.8175,1129.87,1005.815,895.67,905.17,913.5238,899.35,904.7925,775.9813,802.8288,826.1213,818.6388,822.4225,
							830.2638,904.4538,884.2513,898.0213,891.6238,920.0113,925.3313,912.6875,885.6475,894.11,915.1,918.5625,920.28,914.0438,920.2413,967.6675,
							987.3463,1007.795,1018.1113,1033.6525,1013.9938,1024.6125,1052.5388,1054.3438,988.9475,993.0788,1010.0025,999.5125,996.8575,1009.2513,
							1009.1163,1034.0825,1053.1225,1058.4063,1052.8175,1083.2413,1123.6563,1122.195,1178.384,1180.918,1151.581,1179.968,1194.278,1190.888,
							1230.016,1260.924,1290.786,1267.68,1277.685,1280.868,1232.431,1150.481,1191.815,1117.025,1181.645,1229.505,1243.242,1246.309,1259.601,
							1172.909,1070.128,970.598,1017.8,1041.343,1115.039,1037.44,1029.95,935.946,964.692,965.229,1040.491,1044.251,1040.392,1037.527,1079.748,
							1089.511,1098.776,1147.631,1143.746,1134.998,1190.597,1193.019,1184.824,1210.055,1213.339,1224.767,1216.505,1178.533,1183.44,1180.699,
							1184.787,1203.731,1217.596,1226.939,1255.403,1257.135,1244.375,1248.2175,1248.325,1263.545,1284.845,1329.19,1320.0513,1327.0388,1347.9588,
							1402.0838,1443.6825,1491.9988,1515.6288,1512.2088,1548.2863,1555.4713,1639.3225,1706.9313,1756.8025,1807.3725,1676.9938,1759.9613,1772.4163,
							1697.3788,1718.2013,1802.1638,1887.3263,1968.1025,2051.735,2055.6175,2139.0275,2291.4775,2476.2963,2357.5038,2247.4825,2106.3075,2207.5775,
							2289.87,2197.2338,2330.2338,2452.1813,2517.4088,2555.6538,2552.8088,2736.595,2914.0825,2694.2188,2825.0313,2826.7,2942.345,3018.545,2682.595,
							2738.9313,2494.485,2456.9238,2528.1025,2663.9975,2576.1713,2641.665,2778.8275,2712.1575,2740.79,2738.225,2619.1188,2594.4538,2485.3588,2593.17,
							2584.5638,2561.5613,2499.9838,2460.2,2529.7838,2581.0663,2625.0713,2629.2725,2619.1125,2521.2363,2579.9338,2525.675,2371.96,2332.1875,2423.16,
							2364.5163,2232.655,1993.2588,1938.9425,2244.265,2327.8975,2294.405,2877.3863,2694.2913,2838.81,2762.6263,2779.0438,2591.2163,2550.18,2697.4725,
							2805.1788,2720.08,2746.3263,2873.8325,2735.5875,2723.58,2814.3613,2883.6813,3301.7588,3255.0025,3431.9688,3453.1625,3377.5438,3445.28,3679.6075,
							3917.6488,4111.1963,4382.7413,4204.43,4425.2963,4316.34,4159.4575,4206.1313,4111.22,4054.9425,4137.6725,4191.2175,4362.475,4408.3238,4387.4613,
							4394.515,4439.6588,4648.1338,4630.7275,4764.8675,4950.7238,4643.975,4631.695,4319.7213,4422.1213,4626.72,4638.0975,4317.5375,4291.88,4191.175,
							4188.845,4148.2675,3874.2588,3226.4125,3686.9,3678.7375,3672.5663,4067.0775,3896.9988,3858.0888,3612.6813,3603.3088,3777.2938,3662.12,3927.4988,
							3895.5125,4208.5613,4185.2925,4164.1038,4353.0475,4394.6388,4404.0975,4320.09,4225.9238,4322.755,4370.245,4437.0338,4596.9625,4772.975,4754.6988,
							4830.7663,5439.1338,5640.1275,5809.6938,5697.3917,5754.2213,5595.235,5572.1988,5699.5838,5984.0863,6013.2288,5984.9563,5895.2988,5518.85,5733.9038,
							5888.145,5767.68,5732.825,6140.5313,6121.8,6447.6675,6750.1725,7030.0025,7161.4513,7386.9988,7382.4488,6958.2113,7118.8013,7458.795,7146.7813,6570.3125,
							6337,5857.3175,6517.6763,6598.7688,7279.0013,7843.9375,7689.9088,7776.94,8033.9363,8238.2025,8095.5938,8230.6925,8002.6413,8201.4613,8763.785,9326.5888,
							9739.055,9908.2288,9816.3475,9916.5363,10859.5625,10895.0138,11180.8875,11616.855,11696.0583,13708.9913,16858.02,16057.145,14913.4038,15036.9563,16699.6775,
							17178.1025,16407.2025,16531.0838,17601.9438,19343.04,19086.6438,18960.5225,17608.35,16454.7225,15561.05,13857.145,14548.71,13975.4363,13917.0275,15745.2575,
							15378.285,14428.76,14427.87,12629.8138,13860.1363,13412.44,14740.7563,15134.6513,15155.2263,16937.1738,17135.8363,16178.495,14970.3575,14439.4738,14890.7225,
							13287.26,13812.715,14188.785,13619.0288,13585.9013,11348.02,11141.2488,11250.6475,11514.925,12759.6413,11522.8588,10772.15,10839.8263,11399.52,11137.2375,
							11090.0638,11407.1538,11694.4675,11158.3938,10034.9975,10166.5063,9052.5763,8827.63,9224.3913,8186.6488,6914.26,7700.3863,7581.8038,8237.2363,8689.8388,
							8556.6125,8070.7963,8891.2125,8516.2438,9477.84,10016.4888,10178.7125,11092.1475,10396.63,11159.7238,11228.2413,10456.1725,9830.4263,10149.4625,9682.3825,
							9586.46,10313.0825,10564.4188,10309.6413,10907.59,11019.5213,11438.6513,11479.7313,11432.9825,10709.5275,9906.8,9299.2838,9237.05,8787.1638,9532.7413,
							9118.2713,9144.1475,8196.8975,8256.9938,8269.3275,7862.1088,8196.0225,8594.1913,8915.9038,8895.4,8712.8913,8918.7438,8535.8938,8449.835,8138.3363,7790.1575,
							7937.205,7086.4875,6844.3213,6926.0175,6816.74,7049.7938,7417.8875,6789.3013,6774.7513,6620.4088,6896.2813,7022.7088,6773.9388,6830.9038,6939.55,7916.3725,
							7889.2313,8003.6813,8357.0375,8051.345,7890.15,8163.69,8273.7413,8863.5025,8917.5963,8792.83,8938.3038,9652.1563,8864.0875,9278.9975,8978.3325,9342.4713,
							9392.0313,9244.3225,9067.715,9219.8638,9734.675,9692.7175,9826.5975,9619.1438,9362.5338,9180.1588,9306,9014.615,8406.175,8467.6563,8688.0288,8675.2063,
							8474.2388,8340.2988,8058.6038,8240.725,8234.1525,8520.8125,8395.2325,7983.5175,7502.5588,7578.6913,7460.6938,7334.1638,7344.9675,7105.6725,7460.5838,7375.67,
							7487.1875,7518.245,7636.1938,7711.365,7490.59,7616.8913,7655.9788,7688.0038,7616.1025};
	
	
	private double[] baseDolar = {3.273,3.263,3.233,3.213,3.206,3.21,3.192,3.215,3.166,3.203,3.223,3.21,3.221,3.211,3.192,3.161,3.165,3.169,3.18,3.16,3.132,3.127,3.148,3.12,3.124,3.118,
								  3.13,3.126,3.119,3.116,3.118,3.1,3.078,3.051,3.095,3.092,3.097,3.082,3.064,3.099,3.098,3.114,3.136,3.111,3.119,3.148,3.174,3.162,3.154,3.164,3.163,3.108,
								  3.108,3.09,3.077,3.094,3.125,3.128,3.126,3.13,3.123,3.125,3.168,3.117,3.123,3.092,3.116,3.13,3.141,3.142,3.146,3.127,3.104,3.096,3.129,3.145,3.125,3.158,
								  3.185,3.176,3.198,3.172,3.149,3.178,3.176,3.194,3.186,3.161,3.156,3.129,3.101,3.092,3.108,3.381,3.288,3.286,3.265,3.263,3.282,3.261,3.271,3.266,3.244,3.231,
								  3.24,3.282,3.282,3.275,3.284,3.274,3.299,3.321,3.284,3.289,3.298,3.315,3.325,3.336,3.334,3.313,3.317,3.303,3.295,3.308,3.302,3.305,3.319,3.306,3.29,3.265,
								  3.252,3.226,3.211,3.19,3.182,3.167,3.153,3.14,3.126,3.146,3.156,3.166,3.151,3.146,3.131,3.116,3.127,3.119,3.122,3.127,3.132,3.146,3.154,3.17,3.189,3.198,
								  3.167,3.16,3.165,3.144,3.154,3.157,3.14,3.146,3.156,3.17,3.164,3.147,3.133,3.139,3.12,3.113,3.091,3.085,3.114,3.134,3.135,3.126,3.124,3.132,3.128,3.135,
								  3.129,3.141,3.167,3.193,3.187,3.168,3.164,3.15,3.132,3.135,3.165,3.177,3.169,3.164,3.157,3.161,3.177,3.167,3.173,3.183,3.2,3.247,3.239,3.244,3.28,3.255,
								  3.277,3.274,3.292,3.285,3.273,3.251,3.252,3.266,3.287,3.283,3.281,3.279,3.262,3.259,3.256,3.237,3.23,3.222,3.223,3.214,3.262,3.264,3.251,3.232,3.235,3.289,
								  3.281,3.285,3.315,3.304,3.333,3.318,3.288,3.288,3.291,3.305,3.321,3.32,3.303,3.308,3.308,3.27,3.254,3.232,3.241,3.236,3.24,3.247,3.23,3.22,3.196,3.222,3.232,
								  3.213,3.209,3.193,3.225,3.197,3.139,3.145,3.166,3.166,3.162,3.173,3.206,3.236,3.261,3.247,3.269,3.282,3.254,3.221,3.238,3.235,3.251,3.256,3.26,3.242,3.235,
								  3.238,3.245,3.262,3.261,3.258,3.225,3.232,3.252,3.25,3.261,3.249,3.258,3.286,3.291,3.291,3.298,3.292,3.303,3.304,3.303,3.326,3.338,3.324,3.31,3.314,3.354,
								  3.32,3.367,3.39,3.42,3.405,3.386,3.411,3.426,3.404,3.384,3.398,3.41,3.442,3.467,3.504,3.498,3.468,3.481,3.542,3.548,3.531,3.546,3.579,3.594,3.557,3.572,
								  3.61,3.675,3.68,3.687,3.75,3.707,3.65,3.651,3.644,3.659,3.709,3.729,3.737,3.741,3.742,3.775,3.819,3.9,3.786,3.691};

	
	public double[] getBase() {
		return base;
	}
	
	public double[] getBaseDolar() {
		return baseDolar;
	}

}




