# Variants : example : B3A2C0(B3-->L3, A-->L2, C--> v-register)
# ORDER    : C = column major, R = row major
# TRANS    : T = transpose   , N = !T

#IMPLEMENTED ALGORITHMS LIST:
#   [*] B3A2C0 : OK
#   [*] A3B2C0 : OK
#   [*] B3C2A0 : OK
#   [*] C3B2A0 : OK
#   [*] A3C2B0 : NOT IMPLEMENTED
#   [*] C3A2B0 : NOT IMPLEMENTED

#------------------------------------------
#| AMD ZEN2 (AVX2, __mm256, 16 Registers) |
#------------------------------------------
#                                         | 
#   Micro-Kernels                         |
#  +------+------+                        |
#  |  MR  |  NR  |                        |
#  +------+------+                        |
#  |  8   |  8   |                        |
#  +------+------+                        |
#  |  8   |  14  |                        |
#  +------+------+                        |
#  |  16  |  6   |                        |
#  +------+------+                        |
#  |  24  |  4   |                        |
#  +------+------+                        |
#  |  32  |  2   |                        |
#  +------+------+                        |
#  |  40  |  2   | [*] Special Case.      |
#  +------+------+                        |
#------------------------------------------


VARIANT=B3A2C0
ORDERA=R  #R
ORDERB=R  #R
ORDERC=R  #R
TRANSA=N  #T
TRANSB=N  #T

TESTING="Y"
VISUAL=0
ALPHA=1.0   
BETA=1.0   

TIMIN=0.0 
TEST=T

#uMicro="MK_BLIS"
uMicro="BASE"
#uMicro="GENERIC"

uk_len_2v=4
uk_len=4

if [ "$ORDERC" = "R" ]; then
  mr=$uk_len
  nr=$uk_len_2v
else
  mr=$uk_len_2v
  nr=$uk_len
fi

kr=1


make MR=$mr NR=$nr KR=$kr MICRO=$uMicro 

. ./small.sh
#. ./medium.sh
#. ./large.sh
#. ./cache_mc.sh
#. ./cmmse.sh
#. ./tvm.sh
#. ./blis_square.sh
#. ./blis_1fix.sh


if [ "$TESTING" = "NO" ]
then
   if [ "$VARIANT" = "B3A2C0" ]
   then
	echo "$VARIANT"
	. ./blis_b3a2c0.sh
   else 
    if [ "$VARIANT" = "A3B2C0" ]
    then
	#echo "$VARIANT"
	. ./blis_a3b2c0.sh
    fi
        if [ "$VARIANT" = "B3C2A0" ]
        then
	    #echo "$VARIANT"
	    . ./blis_b3c2a0.sh
        fi
            if [ "$VARIANT" = "C3B2A0" ]
            then
	        #echo "$VARIANT"
	        . ./blis_c3b2a0.sh
            fi
   fi
fi

./test_gemm.x $VARIANT $ORDERA $ORDERB $ORDERC $TRANSA $TRANSB $ALPHA $BETA $MMIN $MMAX $MSTEP $NMIN $NMAX $NSTEP $KMIN $KMAX $KSTEP $MCMIN $MCMAX $MCSTEP $NCMIN $NCMAX $NCSTEP $KCMIN $KCMAX $KCSTEP $VISUAL $TIMIN $TEST $1 $2


