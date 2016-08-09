{-# LANGUAGE ForeignFunctionInterface #-}
module Safe where

import Foreign
import Foreign.C.Types
import Data.Vector.Storable
import qualified Data.Vector.Storable as V
--import Network
import AI.HNN.Recurrent.Network
import AI.HNN.FF.Network
import Numeric.LinearAlgebra
import Data.List.Split
import Data.Matrix
import Data.Vector
import Data.List.Split

foreign export ccall process_network_input :: Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr (Ptr Double) -> IO ()
foreign export ccall train :: Ptr CInt -> IO ()

get1st (a,_) = a
get2nd (_,a) = a

concatr :: Double -> [[Double]] -> [[Double]]
concatr x ys = Prelude.map (x:) (rev ys)

foo xs = [(Prelude.length xs-1), (Prelude.length xs -2)..0]
rev xs = [xs !! k| k <- foo xs]

train :: Ptr CInt -> IO ()
train n = do 
    net <- (AI.HNN.FF.Network.createNetwork 50 [50] 6) :: IO (AI.HNN.FF.Network.Network Double)
    let samples = [ Numeric.LinearAlgebra.fromList [0.513672,0.507864,0.543385,0.847786,0.700981,0.741307,0.683966,0.776259,0.001,0.994232,0.50262,0.690233,0.669329,0.747026,0.597004,0.500611,0.651603,0.705245,0.990772,0.504465,0.741317,0.86633,0.626058,0.50003,0.62625,0.912152,0.951031,0.695062,0.988584,0.500696,0.924951,0.793749,0.735071,0.001,0.898172,0.741072,0.557822,0.500193,0.531111,0.975972,0.992257,0.808622,0.659775,0.780786,0.989029,0.649204,0.573373,0.981073,0.930637,0.523214] --> Numeric.LinearAlgebra.fromList [0, 1, 0, 0, 0, 0]
		, Numeric.LinearAlgebra.fromList [0.001,0.500812,0.506793,0.912596,0.654375,0.731504,0.531926,0.571592,0.001,0.99705,0.001,0.53954,0.503728,0.932112,0.501825,0.001,0.516957,0.544514,0.994694,0.001,0.606682,0.72213,0.502379,0.001,0.526345,0.992499,0.970231,0.549113,0.989124,0.001,0.830758,0.594613,0.552365,0.001,0.774369,0.570688,0.501079,0.50002,0.001,0.976555,0.97934,0.589607,0.517316,0.59639,0.987111,0.50283,0.511854,0.954333,0.864359,0.001] --> Numeric.LinearAlgebra.fromList [1, 0, 0, 0, 0, 0]
		]
    let smartNet = trainNTimes 100000 0.01 AI.HNN.FF.Network.tanh tanh' net samples
    --print smartNet
    saveNetwork "smartNet.nn" smartNet

feed2 :: Int -> [Double] -> [Double] -> [Double -> Double] -> IO (Data.Vector.Storable.Vector Double)
feed2 nodes_number weights inputs_ functions = do
    --let weights_layers = Prelude.splitAt 72 weights
    --let weights_list = Prelude.concat (concatr 0.0 (splitEvery 6 (get1st weights_layers)))
    --let input_layer__ = (12><7) weights_list
    --let hidden_layer__ = (4><12) (get2nd weights_layers)
    --let vector_ = Data.Vector.fromList [input_layer__, hidden_layer__] --also had 'output_layer__' before
    --let n = AI.HNN.FF.Network.fromWeightMatrices vector_ :: AI.HNN.FF.Network.Network Double
    --let inputs__ = Numeric.LinearAlgebra.fromList inputs_
    --return (output n AI.HNN.FF.Network.tanh inputs__)
    n <- AI.HNN.FF.Network.loadNetwork "smartNet.nn" :: IO (AI.HNN.FF.Network.Network Double)
    --print n
    let inputs__ = Numeric.LinearAlgebra.fromList inputs_
    return (output n AI.HNN.FF.Network.tanh inputs__)


feed :: Int -> [Double] -> [[Double]] -> [Double -> Double] -> IO (Data.Vector.Storable.Vector Double)
feed nodes_number weights inputs_ functions = do
    let numNeurons = 101
        numInputs  = 100
        thresholds = Prelude.replicate numNeurons 0.0
        inputs     = inputs_
        adj        = weights
    n <- AI.HNN.Recurrent.Network.createNetwork numNeurons numInputs adj thresholds :: IO (AI.HNN.Recurrent.Network.Network Double)
    output <- evalNet n inputs AI.HNN.FF.Network.tanh
    return output

peekInt :: Ptr CInt -> IO Int
peekInt = fmap fromIntegral . peek

build_function_list :: [Double] -> [Double -> Double]
build_function_list (x:xs) | x == 0.0 = AI.HNN.FF.Network.tanh : build_function_list xs
                           | otherwise = AI.HNN.FF.Network.tanh : build_function_list xs

process_network_input :: Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr (Ptr Double) -> IO ()
process_network_input atype n m nodes_number weights inputs functions result = do
    atype <- peekInt atype
    n <- peekInt n
    m <- peekInt m
    nodes_number <- peekInt nodes_number
    weights_ <- peekArray n weights
    --print weights_
    inputs_ <- peekArray m inputs
    let inputs__ = inputs_:[]
    --print inputs__
    functions_ <- peekArray nodes_number functions
    let functions_list = build_function_list functions_
    res <- case (atype == 0) of
        True -> (feed (Prelude.length functions_) weights_ inputs__ functions_list) --used inputs__ before
        False -> (feed2 (Prelude.length functions_) weights_ inputs_ functions_list) --used inputs__ before
    --print res
    let aList = (V.toList res)
    let b = (fromIntegral (Prelude.length aList)):aList
    ptr <- newArray b
    poke result $ ptr
