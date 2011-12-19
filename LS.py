# Python code
# Author: Bruno Turcksin
# Date: 2011-12-13 15:54:23.139234

#----------------------------------------------------------------------------#
## Class LS                                                                 ##
#----------------------------------------------------------------------------#

"""Contain the Level Symmetric quadrature"""

import numpy as np
import QUADRATURE as QUAD
import utils

class LS(QUAD.QUADRATURE) :
  """Build the Level Symmetric quadrature up to order 12."""

  def __init__(self,sn,L_max,galerkin) :

    super(LS,self).__init__(sn,L_max,galerkin)

#----------------------------------------------------------------------------#

  def Build_quadrant(self) :
    """Build omega and weight for one quadrant."""

    self.omega = np.zeros((self.n_dir,3))
    self.weight = np.zeros((self.n_dir))

    if self.sn==2 :
      direction = 0.577350269189625764509149
      weight = 1.

      self.omega[0,0] = direction
      self.omega[0,1] = direction
      self.omega[0,2] = direction
    
      self.weight[0] = weight
    
    elif self.sn==4 :
      direction_1 = 0.350021174581540677777041
      direction_2 = 0.868890300722201205229788
      weight = 1./3.

      self.omega[0,0] = direction_2
      self.omega[0,1] = direction_1
      self.omega[0,2] = direction_1
      
      self.omega[1,0] = direction_1
      self.omega[1,1] = direction_2
      self.omega[1,2] = direction_1

      self.omega[2,0] = direction_1
      self.omega[2,1] = direction_1
      self.omega[2,2] = direction_2

      self.weight[0] = weight
      self.weight[1] = weight
      self.weight[2] = weight

    elif self.sn==6 :
      direction_1 = 0.266635401516704720331535
      direction_2 = 0.681507726536546927403750
      direction_3 = 0.926180935517489107558380
      weight_1 = 0.176126130863383433783565
      weight_2 = 0.157207202469949899549768

      self.omega[0,0] = direction_3
      self.omega[0,1] = direction_1
      self.omega[0,2] = direction_1

      self.omega[1,0] = direction_2
      self.omega[1,1] = direction_2
      self.omega[1,2] = direction_1

      self.omega[2,0] = direction_1
      self.omega[2,1] = direction_3
      self.omega[2,2] = direction_1

      self.omega[3,0] = direction_2
      self.omega[3,1] = direction_1
      self.omega[3,2] = direction_2
      
      self.omega[4,0] = direction_1
      self.omega[4,1] = direction_2
      self.omega[4,2] = direction_2

      self.omega[5,0] = direction_1
      self.omega[5,1] = direction_1
      self.omega[5,2] = direction_3

      self.weight[0] = weight_1
      self.weight[1] = weight_2
      self.weight[2] = weight_1
      self.weight[3] = weight_2
      self.weight[4] = weight_2
      self.weight[5] = weight_1

    elif self.sn==8 :
      direction_1 = 0.218217890235992381266097
      direction_2 = 0.577350269189625764509149
      direction_3 = 0.786795792469443145800830
      direction_4 = 0.951189731211341853132399

      weight_1 = 0.120987654320987654320988
      weight_2 = 0.0907407407407407407407407
      weight_3 = 0.0925925925925925925925926

      self.omega[0,0] = direction_4
      self.omega[0,1] = direction_1
      self.omega[0,2] = direction_1

      self.omega[1,0] = direction_3
      self.omega[1,1] = direction_2
      self.omega[1,2] = direction_1
      
      self.omega[2,0] = direction_2
      self.omega[2,1] = direction_3
      self.omega[2,2] = direction_1

      self.omega[3,0] = direction_1
      self.omega[3,1] = direction_4
      self.omega[3,2] = direction_1

      self.omega[4,0] = direction_3
      self.omega[4,1] = direction_1
      self.omega[4,2] = direction_2

      self.omega[5,0] = direction_2
      self.omega[5,1] = direction_2
      self.omega[5,2] = direction_2

      self.omega[6,0] = direction_1
      self.omega[6,1] = direction_3
      self.omega[6,2] = direction_2

      self.omega[7,0] = direction_2
      self.omega[7,1] = direction_1
      self.omega[7,2] = direction_3

      self.omega[8,0] = direction_1
      self.omega[8,1] = direction_2
      self.omega[8,2] = direction_3

      self.omega[9,0] = direction_1
      self.omega[9,1] = direction_1
      self.omega[9,2] = direction_4

      self.weight[0] = weight_1
      self.weight[1] = weight_2
      self.weight[2] = weight_2
      self.weight[3] = weight_1
      self.weight[4] = weight_2
      self.weight[5] = weight_3
      self.weight[6] = weight_2
      self.weight[7] = weight_2
      self.weight[8] = weight_2
      self.weight[9] = weight_1

    elif self.sn==10 :
      direction_1 = 0.189321326478010476671494
      direction_2 = 0.508881755582618974382711
      direction_3 = 0.694318887594384317279217
      direction_4 = 0.839759962236684758403029
      direction_5 = 0.963490981110468484701598

      weight_1 = 0.0893031479843567214704325
      weight_2 = 0.0725291517123655242296233
      weight_3 = 0.0450437674364086390490892
      weight_4 = 0.0539281144878369243545650

      self.omega[0,0] = direction_5
      self.omega[0,1] = direction_1
      self.omega[0,2] = direction_1
      
      self.omega[1,0] = direction_4
      self.omega[1,1] = direction_2
      self.omega[1,2] = direction_1
      
      self.omega[2,0] = direction_3
      self.omega[2,1] = direction_3
      self.omega[2,2] = direction_1
      
      self.omega[3,0] = direction_2
      self.omega[3,1] = direction_4
      self.omega[3,2] = direction_1

      self.omega[4,0] = direction_1
      self.omega[4,1] = direction_5
      self.omega[4,2] = direction_1

      self.omega[5,0] = direction_4
      self.omega[5,1] = direction_1
      self.omega[5,2] = direction_2

      self.omega[6,0] = direction_3
      self.omega[6,1] = direction_2
      self.omega[6,2] = direction_2

      self.omega[7,0] = direction_2
      self.omega[7,1] = direction_3
      self.omega[7,2] = direction_2

      self.omega[8,0] = direction_1
      self.omega[8,1] = direction_4
      self.omega[8,2] = direction_2

      self.omega[9,0] = direction_3
      self.omega[9,1] = direction_1
      self.omega[9,2] = direction_3

      self.omega[10,0] = direction_2
      self.omega[10,1] = direction_2
      self.omega[10,2] = direction_3

      self.omega[11,0] = direction_1
      self.omega[11,1] = direction_3
      self.omega[11,2] = direction_3

      self.omega[12,0] = direction_2
      self.omega[12,1] = direction_1
      self.omega[12,2] = direction_4

      self.omega[13,0] = direction_1
      self.omega[13,1] = direction_2
      self.omega[13,2] = direction_4

      self.weight[0] = weight_1
      self.weight[1] = weight_2
      self.weight[2] = weight_3
      self.weight[3] = weight_2
      self.weight[4] = weight_1
      self.weight[5] = weight_2
      self.weight[6] = weight_4
      self.weight[7] = weight_4
      self.weight[8] = weight_2
      self.weight[9] = weight_3
      self.weight[10] = weight_4
      self.weight[11] = weight_3
      self.weight[12] = weight_2
      self.weight[13] = weight_2
      self.weight[14] = weight_1

    elif self.sn==12 :
      direction = np.zeros((6,1))

      direction[0] = 0.167212652822713264084504
      direction[1] = 0.459547634642594690016761
      direction[2] = 0.628019096642130901034766
      direction[3] = 0.760021014833664062877138
      direction[4] = 0.872270543025721502340662
      direction[5] = 0.971637719251358378302376

      weight_1 = 0.0707625899700910439766549
      weight_2 = 0.0558811015648888075828962
      weight_3 = 0.0373376737588285824652402
      weight_4 = 0.0502819010600571181385765
      weight_5 = 0.0258512916557503911218290

      for i in xrange(0,6) :
        self.omega[i,0] = direction[5-i]
        self.omega[i,1] = direction[i]
        self.omega[i,2] = direction[0]
      
      offset = 6
      for i in xrange(0,5) :
        self.omega[offset+i,0] = direction[4-i]
        self.omega[offset+i,1] = direction[i]
        self.omega[offset+i,2] = direction[1]

      offset += 5
      for i in xrange(0,4) :
        self.omega[offset+i,0] = direction[3-i]
        self.omega[offset+i,1] = direction[i]
        self.omega[offset+i,2] = direction[2]
       
      offset += 4
      for i in xrange(0,3) :
        self.omega[offset+i,0] = direction[2-i]
        self.omega[offset+i,1] = direction[i]
        self.omega[offset+i,2] = direction[3]

      offset += 3
      for i in xrange(0,2) :
        self.omega[offset+i,0] = direction[1-i]
        self.omega[offset+i,1] = direction[i]
        self.omega[offset+i,2] = direction[4]
      
      offset += 2
      self.omega[offset+i,0] = direction[0]
      self.omega[offset+i,1] = direction[1]
      self.omega[offset+i,2] = direction[5]

      self.weight[0] = weigth_1
      self.weight[1] = weight_2
      self.weight[2] = weight_3
      self.weight[3] = weight_3
      self.weight[4] = weight_2
      self.weight[5] = weight_1
      self.weight[6] = weight_2
      self.weight[7] = weight_4
      self.weight[8] = weight_5
      self.weight[9] = weight_4
      self.weight[10] = weight_2
      self.weight[11] = weight_3
      self.weight[12] = weight_5
      self.weight[13] = weight_5
      self.weight[14] = weight_3
      self.weight[15] = weight_3
      self.weight[16] = weight_4
      self.weight[17] = weight_3
      self.weight[18] = weight_2
      self.weight[19] = weight_2
      self.weight[20] = weight_1

## The quadrature is wrong for higher order

#    elif self.sn==14 :
#      direction = np.zeros((7,1))
#
#      direction[0] = 0.151985861461031912404799
#      direction[1] = 0.422156982304796966896263
#      direction[2] = 0.577350269189625764509149
#      direction[3] = 0.698892086775901338963210
#      direction[4] = 0.802226255231412057244328
#      direction[5] = 0.893691098874356784901111
#      direction[6] = 0.976627152925770351762946
#
#      weight_1 = 0.0579970408969969964063611
#      weight_2 = 0.0489007976368104874582568
#      weight_3 = 0.0227935342411872473257345
#      weight_4 = 0.0394132005950078294492985
#      weight_5 = 0.0380990861440121712365891
#      weight_6 = 0.0258394076418900119611012
#      weight_7 = 0.00826957997262252825269908
#
#      offset += 0
#      for j in xrange(0,7) :
#        for i in xrange(0,7-j) :
#          self.omega[offset+i,0] = direction[(6-j)-i]
#          self.omega[offset+i,1] = direction[i]
#          self.omega[offset+i,2] = direction[j]
#        offset += 6-j
#      
#      self.weight[0] = weight_1
#      self.weight[1] = weight_2
#      self.weight[2] = weight_3
#      self.weight[3] = weight_4
#      self.weight[4] = weight_3
#      self.weight[5] = weight_2
#      self.weight[6] = weight_1
#      self.weight[7] = weight_2
#      self.weight[8] = weight_5
#      self.weight[9] = weight_6
#      self.weight[10] = weight_6
#      self.weight[11] = weight_5
#      self.weight[12] = weight_2
#      self.weight[13] = weight_3 
#      self.weight[14] = weight_6
#      self.weight[15] = weight_7
#      self.weight[16] = weight_6
#      self.weight[17] = weight_3
#      self.weight[18] = weight_4
#      self.weight[19] = weight_6
#      self.weight[20] = weight_6
#      self.weight[21] = weight_4
#      self.weight[22] = weight_3
#      self.weight[23] = weight_5
#      self.weight[24] = weight_3
#      self.weight[25] = weight_2
#      self.weight[26] = weight_2
#      self.weight[27] = weight_1
#
#    elif self.sn==16 :
#      direction = np.zeros((8,1))
#
#      direction[0] = 0.138956875067780344591732
#      direction[1] = 0.392289261444811712294197
#      direction[2] = 0.537096561300879079878296
#      direction[3] = 0.650426450628771770509703
#      direction[4] = 0.746750573614681064580018
#      direction[5] = 0.831996556910044145168291
#      direction[6] = 0.909285500943725291652116
#      direction[7] = 0.980500879011739882135849
#
#      weight_1 = 0.0489872391580385335008367
#      weight_2 = 0.0413295978698440232405505
#      weight_3 = 0.0203032007393652080748070
#      weight_4 = 0.0265500757813498446015484
#      weight_5 = 0.0379074407956004002099321
#      weight_6 = 0.0135295047786756344371600
#      weight_7 = 0.0326369372026850701318409
#      weight_8 = 0.0103769578385399087825920
#
#      offset += 0
#      for j in xrange(0,8) :
#        for i in xrange(0,8-j) :
#          self.omega[offset+i,0] = direction[(7-j)-i]
#          self.omega[offset+i,1] = direction[i]
#          self.omega[offset+i,2] = direction[j]
#        offset += 7-j
#
#      self.weight[0] = weight_1 
#      self.weight[1] = weight_2
#      self.weight[2] = weight_3
#      self.weight[3] = weight_4
#      self.weight[4] = weight_4
#      self.weight[5] = weight_3
#      self.weight[6] = weight_2
#      self.weight[7] = weight_1
#      self.weight[8] = weight_2
#      self.weight[9] = weight_5
#      self.weight[10] = weight_6
#      self.weight[11] = weight_7
#      self.weight[12] = weight_6
#      self.weight[13] = weight_5
#      self.weight[14] = weight_2
#      self.weight[15] = weight_3
#      self.weight[16] = weight_6
#      self.weight[17] = weight_8
#      self.weight[18] = weight_8
#      self.weight[19] = weight_6
#      self.weight[20] = weight_3
#      self.weight[21] = weight_4
#      self.weight[22] = weight_7
#      self.weight[23] = weight_8
#      self.weight[24] = weight_7
#      self.weight[25] = weight_4
#      self.weight[26] = weight_4
#      self.weight[27] = weight_6
#      self.weight[28] = weight_6
#      self.weight[29] = weight_4
#      self.weight[30] = weight_3
#      self.weight[31] = weight_5
#      self.weight[32] = weight_3
#      self.weight[33] = weight_2
#      self.weight[34] = weight_2
#      self.weight[35] = weight_1
#
#    elif self.sn==18 :
#      direction = np.zeros((9,1))
#
#      direction[0] = 0.129344504545924818514086
#      direction[1] = 0.368043816053393605686086
#      direction[2] = 0.504165151725164054411848
#      direction[3] = 0.610662549934881101060239
#      direction[4] = 0.701166884252161909657019
#      direction[5] = 0.781256199495913171286914
#      direction[6] = 0.853866206691488372341858
#      direction[7] = 0.920768021061018932899055
#      direction[8] = 0.983127661236087115272518
#
#      weight_1 = 0.0422646448843821748535825
#      weight_2 = 0.0376127473827281471532380
#      weight_3 = 0.0122691351637405931037187
#      weight_4 = 0.0324188352558815048715646
#      weight_5 = 0.00664438614619073823264082
#      weight_6 = 0.0312093838436551370068864
#      weight_7 = 0.0160127252691940275641645
#      weight_8 = 0.0200484595308572875885066
#      weight_9 = 0.000111409402059638628382279
#      weight_10 = 0.0163797038522425240494567
#
#      offset += 0
#      for j in xrange(0,9) :
#        for i in xrange(0,9-j) :
#          self.omega[offset+i,0] = direction[(8-j)-i]
#          self.omega[offset+i,1] = direction[i]
#          self.omega[offset+i,2] = direction[j]
#        offset += 8-j
#      
#      self.weight[0] = weight_1
#      self.weight[1] = weight_2
#      self.weight[2] = weight_3
#      self.weight[3] = weight_4
#      self.weight[4] = weight_5
#      self.weight[5] = weight_4
#      self.weight[6] = weight_3
#      self.weight[7] = weight_2
#      self.weight[8] = weight_1
#      self.weight[9] = weight_2
#      self.weight[10] = weight_6
#      self.weight[11] = weight_7
#      self.weight[12] = weight_8
#      self.weight[13] = weight_8
#      self.weight[14] = weight_7
#      self.weight[15] = weight_6
#      self.weight[16] = weight_2
#      self.weight[17] = weight_7
#      self.weight[18] = weight_9
#      self.weight[19] = weight_10
#      self.weight[20] = weight_9
#      self.weight[21] = weight_7
#      self.weight[22] = weight_3
#      self.weight[23] = weight_4
#      self.weight[24] = weight_8
#      self.weight[25] = weight_10
#      self.weight[26] = weight_10
#      self.weight[27] = weight_8
#      self.weight[28] = weight_4
#      self.weight[29] = weight_5
#      self.weight[30] = weight_8
#      self.weight[31] = weight_9
#      self.weight[32] = weight_8
#      self.weight[33] = weight_5
#      self.weight[34] = weight_4
#      self.weight[35] = weight_7
#      self.weight[36] = weight_7
#      self.weight[37] = weight_4
#      self.weight[38] = weight_3
#      self.weight[39] = weight_6
#      self.weight[40] = weight_3
#      self.weight[41] = weight_2
#      self.weight[42] = weight_2
#      self.weight[43] = weight_1
#
#    elif self.sn==20 :
#      direction = np.zeros((10,1))
#
#      direction[0] = 0.120603343036693597409418
#      direction[1] = 0.347574292315847257336779
#      direction[2] = 0.476519266143665680817278
#      direction[3] = 0.577350269489625764509149
#      direction[4] = 0.663020403653288019308789
#      direction[5] = 0.738822561910371432904974
#      direction[6] = 0.807540401661143067193530
#      direction[7] = 0.870852583760463975580977
#      direction[8] = 0.929863938955324566667817
#      direction[9] = 0.985347485558646574628509
#
#      weight_1 =  
#      weight_2 = 
#      weight_3 = 
#      weight_4 = 
#      weight_5 = 
#      weight_6 = 
#      weight_7 = 
#      weight_8 = 
#      weight_9 = 
#      weight_10 = 
#
#      offset += 0
#      for j in xrange(0,10) :
#        for i in xrange(0,10-j) :
#          self.omega[offset+i,0] = direction[(9-j)-i]
#          self.omega[offset+i,1] = direction[i]
#          self.omega[offset+i,2] = direction[j]
#        offset += 9-j
#
