import sys, os
import math,functools

## command line options parser
#__________________________________________________________________________________________________
def proc_cmdline():
  from optparse import OptionParser
  curdir = os.getcwd()
  usage = "%prog [options] arg"
  parser = OptionParser(usage)
  #parser.add_option("-t", "--totMB", type="int", dest="totMegs", default=1024, help="total backup size in MB, default=%default MB")
  #parser.add_option("-n", "--nFiles", type="int", dest="numFiles", default=1024, help="number of files per backup, default=%default")
  #parser.add_option("-f", "--file", type="string", dest="destFile", help="file to store hash values [default random file name, ext: .hash]")
  #parser.add_option("-d", "--dir", type="string", dest="destDir", help="destination directory of hash file [default = \"./\"]")

  (options, args) = parser.parse_args()
  usr_args = []
  joboptions = []
  if len(args) > 0:
    for arg in args:
      if arg[0] != "-":
        joboptions.append(arg)
      else:
        usr_args.append(arg)
  return (options,args,parser)

#__________________________________________________________________________________________________
def isPrime( num ):
  if num < 2:
    return False

  if num == 2:
    return True

  if num % 2 == 0:
    return False
  
  for i in range(2, int(math.sqrt(num))+1 ):
    if num % i == 0:
      return False

  return True

#__________________________________________________________________________________________________
def highestPowerOf(numr,denr):
  if denr == 0 or (denr > numr):
    return numr
  i = 0
  fac = numr
  while True:
    fac = int(math.pow(denr,i))
    if numr % fac == 0:
      i += 1
    else:
      i -= 1
      break
  return i, int((numr / int(math.pow(denr,i))))

#__________________________________________________________________________________________________
def inumPalindrome(num):
  if num < 10:
    return False
  
  digits = []
  while num > 0:
    digits.append( num - int(num/10)*10)
    num = int(num / 10)
  last = int(len(digits)/2)
  for i in range(0,last):
    if not (digits[i] == digits[len(digits)-i-1]):
      return False
  return True

#__________________________________________________________________________________________________
def istrPalindrome(data):
  lc = len(data)
  last = int(lc/2)
  for i in range(0,last):
    if not (data[i] == data[lc-i-1]):
        return False
  return True

#__________________________________________________________________________________________________
def GCD(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a
#__________________________________________________________________________________________________
def LCM(a, b):
    """Return lowest common multiple."""
    return a * b // GCD(a, b)
#__________________________________________________________________________________________________
def LCMultiples(*args):
    """Return lcm of args."""   
    return functools.reduce(LCM, args)

#__________________________________________________________________________________________________
def SumNSquare(num):
  return int (num * (num + 1) * (2*num + 1) / 6)
#__________________________________________________________________________________________________
def SumN(num):
  return int (num * (num +1) / 2)

#__________________________________________________________________________________________________
def PythagoreanTriple(a,b):
  return (a*a + b*b - (1000-a-b)*(1000-a-b) == 0)

#__________________________________________________________________________________________________
def readGrid(filen, delimiter, doprn = False):
  rows = None
  cols = None
  grid = ''
  lines = [line.strip() for line in open(filen)]
  data = []
  rows = len(lines)
  for line in lines:
    row = line.split(delimiter)
    cols = len(row)
    data.append (row)
    if doprn:
      grid += (' '.join(map(str, row)))
      grid += '\n'
  return (rows, cols, data,grid)
  
#__________________________________________________________________________________________________
def LargestAdjProduct(data):
  rows = len(data)
  cols = len(data[0])
  products = [1 for x in range(rows + cols + 2)]
  for i in range(0,cols):
    for j in range(0,rows):
      num = int(data[i][j])
      if i==j:
        products[rows+cols] *= num
      if (i+j) == (rows-1):
        products[rows+cols+1] *= num
      products[j+rows] *= num;
      products[i] *= num
  return (sorted(products)[rows+cols+1])
 
#__________________________________________________________________________________________________
def Divisors(num):
  divisors = []
  for i in range(1,num):
    if num % i == 0:
      divisors.append(i)
  divisors.append(num)
  return divisors

#__________________________________________________________________________________________________
def ProperDivisors(num):
  propdiv = []
  p = ListOfPrimesBelow(int(math.sqrt(num)))
  for i in p:
    f = 0
    while (num % i == 0):
      num = int(num/i)
      f += 1
    if f > 0:
      propdiv.append([i,f])
  if num > 0:
    propdiv.append([num,1])
  return propdiv

#__________________________________________________________________________________________________
def SumDivisors(num):
  d = ProperDivisors(num)
  s = 1
  for i in d:
    if i[0] > 1:
      s *= ( pow(i[0],(i[1]+1))-1)/(i[0]-1)
  return (s - num)

#__________________________________________________________________________________________________
def TriangleNumber(rank):
  if rank == 0:
    return 0
  return SumN(rank)

#__________________________________________________________________________________________________
def CollatzSequence(num):
  colseq = []
  while num > 1:
    colseq.append(num)
    if num % 2 == 0:
      num = int(num / 2)
    else:
      num = 3 * num + 1

  colseq.append(1)
  return colseq
#__________________________________________________________________________________________________
def PlaceValue(num):
  pvl = []
  if num > 100:
    val = int(num/100) * 100
    pvl.append(val)
    rem = num % val
    num  = num % val
  if num > 20:
    val = int(num /10) * 10
    pvl.append (val)
    rem = num % val
    if rem > 0:
      pvl.append(rem)
  else:
    if num > 0:
      pvl.append(num)
  return pvl


#__________________________________________________________________________________________________
def SumPrimes( plist ):
  sumpr = 0
  for i in plist:
    sumpr = sumpr + i
  return sumpr

#__________________________________________________________________________________________________
def ListOfPrimesBelow(num):
  p = 3 ## prime 
  lp = [2]
  while p < num:
    if isPrime(p):
      lp.append(p) 
    p += 2
  return lp

#__________________________________________________________________________________________________
def ListOfPerfectSquares(num):
  p = 1  
  lp = []
  while p < num:
    lp.append(p*p) 
    p += 1
  return lp

#__________________________________________________________________________________________________
def GoldBachConj(num, lp, doprnt = False):
  pq = ListOfPerfectSquares(num)
  
  for i in lp:
    if i > num:
      break
    delta = int(num - i) 
    if not (delta % 2 == 0):
      continue
    delta = delta / 2
    if delta in pq:
      return True
  return False

#__________________________________________________________________________________________________
def Base10ToBase2(num):
  res=""
  while(num > 0):
    app="1"
    bit = num % 2
    if bit == 0:
      app = "0"
    res = app + res
    num = int(num/2)
  return res

#__________________________________________________________________________________________________
def CircularNum(num):
  data = []
  data.append(num)
  if num < 11:
    return data
  
  p = int(math.pow(10,len(str(num))-1))
  while True:
    nnext = int(num / 10) + p * (num % 10)
    if nnext % 2 == 0:
      return None
    if nnext == data[0]:
      break
    data.append(nnext)
    num = nnext
  return data

#__________________________________________________________________________________________________
def DigitsNum(num):
  data = []
  while num >= 10:
    data.append(num % 10)
    num = int(num / 10)
  if num < 10:
    data.append(num)
  return data


#__________________________________________________________________________________________________
def PermuteDigits(data, low=0):
  if low + 1 >= len(data):
    yield data
  else:
    for p in PermuteDigits(data, low + 1):
      yield p        
      for i in range(low + 1, len(data)):        
        data[low], data[i] = data[i], data[low]
        for p in PermuteDigits(data, low + 1):
          yield p        
        data[low], data[i] = data[i], data[low]

#__________________________________________________________________________________________________
def ReverseNum(num):
  if num < 10: 
    return num
  k = 0
  arr = DigitsNum(num)
  rev = 0
  for i in range(0,len(arr)):
    rev = 10*rev + arr[i]
    k = k + 1
  return rev

#__________________________________________________________________________________________________
def Factorial(num):
  res = 1
  for i in range(1,num+1):
    res = i * res;
  return res

#__________________________________________________________________________________________________
def Problem0011(filen):
 
  outp = """11) Largest product in a grid: What is the greatest product of four adjacent
numbers in the same direction (up, down, left, right, or diagonally) in the 20x20 grid?\n"""
  rows, cols, data, grid = readGrid(filen, " ", True)
  outp += grid
  largenum = 0
  largegrid = None
  for chunk in range(0,int(cols-4+1)):
    for j in range (0, int(rows-4+1)):
      m4by4 = []
      for i in range(chunk, chunk+4):
        v1by4 = []
        for k in range(0, 4):
          v1by4.append(data[i][j+k])
        m4by4.append(v1by4)
      product = LargestAdjProduct(m4by4)
      if product > largenum:
        largenum = product
        largegrid = m4by4
      #print (largenum)
      #sys.stdin.read(1)
        
  outp += "\nAnswer: " + str(largenum) + '\n'
  for i in largegrid:
    outp += (' '.join(map(str, i)))
    outp += '\n'
  print (outp,"\n========================================================")
  
#__________________________________________________________________________________________________
def Problem0012(filen):
 
  outp = """12) Highly divisible triangular number: What is the value of the first triangle 
  number to have over five hundred divisors?"""
  #i = 12375
  i = 12370
  outf = open(filen, 'w')
  while True:
    trinum = TriangleNumber(i) 
    numd = len(Divisors(trinum))
    print ("%10d  %10d   %10d" %(i,trinum, numd), end="\n", file=outf)
    if numd > 500:
      print ("%10d  %10d   %10d" % (i,trinum, numd), end="\n", file=outf)
      break
    i += 1
  outf.close()
  outp += "\nAnswer: " + str(i) + '\n'
  print (outp,"\n========================================================")


#__________________________________________________________________________________________________
def Problem0014():
  outp = """14) Longest Collatz sequence: Which starting number, under one million, 
  produces the longest chain? """
  longcolseq = 0
  longseed = 0
  colseq = []
  #for i in range(1000000,13,-1):
  for i in range(837799,837798,-1):
    colseq = CollatzSequence(i)
    if len(colseq) > longcolseq:
      longcolseq = len(colseq)
      longseed = i

  outp += "\nAnswer: " + str(longseed) + '\n'
  print (outp,"\n========================================================")

 
#__________________________________________________________________________________________________
def Problem0017(filen):

  outp = """14) If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, 
                how many letters would be used? """
  n2w={}
  lines = [line.strip() for line in open(filen)]
  for line in lines:
    n2w[line.split()[0]] = line.split()[1]
  
  countwords=0
  for item in range(1,1000):
    word = ""
    numpv = PlaceValue(item)
    nitems = len(numpv)
    val = int(numpv[0] / 100)
    if val >= 1:
      word += n2w[str (val)] + n2w ["100"]
    else:
      word += n2w[str(numpv[0])]
    if nitems > 1:
      if numpv[0] >= 100:
        word += "and"
      word += n2w [ str(numpv[1]) ]
    if nitems > 2:
      word += n2w [ str(numpv[2]) ]
    countwords += len(word)
  print (n2w["1"], n2w["1000"])
  countwords += len(n2w["1"]) + len(n2w["1000"])
  
  outp += "\nAnswer: " + str(countwords) + '\n'
  print (outp,"\n========================================================")


#__________________________________________________________________________________________________
def Problem0018():
  numbers = [ 
      75,
      95, 64,
      17, 47, 82,
      18, 35, 87, 10,
      20,4,82,47,65,
      19,1,23,75,3,34,
      88,2,77,73,7,63,67,
      99,65,4,28,6,16,70,92,
      41,41,26,56,83,40,80,70,33,
      41,48,72,33,47,32,37,16,94,29,
      53,71,44,65,25,43,91,52,97,51,14,
      70,11,33,28,77,73,17,78,39,68,17,57,
      91,71,52,38,17,14,91,43,58,50,27,29,48,
      63,66,4,68,89,53,67,30,73,16,69,87,40,31,
      4,62,98,27,23,9,70,98,73,93,38,53,60,4,23 ]
  
  nr = 15; ## number of rows

  ## this part reads in all the rows
  j = 1; i = 1; rows = []
  deltas = [ i  for i in range(0,nr+2)]
  while True:
    if j == len(deltas)-1:
      break
    trow = []
    for k in range(i, i+deltas[j]):
      trow.append(numbers[k-1])
    j += 1
    i = i+deltas[j]-1
    rows.append(trow)

  ## now search for the path
  path = rows[0]
  for i in range(0,len(rows)-1):
    d = min(rows[i])
    ranks = [ i-d for i in rows[i] ]
    print (ranks)
    path.append(ranks[len(ranks)-1])
  print (path, sum(path))
    

#__________________________________________________________________________________________________
def Problem0020():
  outp = """Find the sum of the digits in the number 100!"""
  data = str(Factorial(100))
  sumi = 0
  for i in list(data):
    sumi = sumi + int(i)
  
  outp += "\nAnswer: " + str(sumi) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0021():
  outp = """Evaluate the sum of all the amicable numbers under 10000. """
  s = 0
  for i in range(2, 10001):
    j = SumDivisors(i)
    if i == SumDivisors(j):
      if i != j:
        s += i
  outp += "\nAnswer: " + str(s) + '\n'
  print (outp,"\n========================================================") 

#__________________________________________________________________________________________________
def Problem0022(filen):
  outp = """What is the total of all the name scores in the file?"""
  lines = [line.strip() for line in open(filen)]  
  data = lines[0].split(",")
  data1 = []
  data1 = [item.replace("\"","") for item in data]
  data = sorted(data1)
  namescore = 0
  for i in range(0, len(data)):
    letters = list(data[i])
    pos = i+1
    sumi = 0
    for l in letters:
      sumi = sumi + (ord(l) - 64)
    #if data[i] == "COLIN":
    #  print (data[i], pos, sumi)
    namescore  = namescore + (pos * sumi)
  outp += "\nAnswer: " + str(namescore) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0030():
  outp = """Find the sum of all the numbers that can be written as the sum of fifth powers of their digits"""
  result = 0
  for i in range(1000,999999):
    digits = DigitsNum(i)
    sum5 = 0
    found = False
    for j in digits:
      sum5 = sum5 + (j*j*j*j*j)
      if sum5 > i:
        break
    if sum5 == i:
      result = result + i
      #print (i)
  outp += "\nAnswer: " + str(result) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0032():
  outp = """Find the sum of all products whose multiplicand/multiplier/product identity can be written as a 1 through 9 pandigital."""
  d = DigitsNum(123456789)
  perm = []
  w = [ str(i) for i in d ]
  sprod = []
  ## get all numbers of form (abcd) : there are 9P4 = 3042 permutations
  for i in w:
    for j in w:
      for k in w:
        for l in w:
          if (i==j) or (i == k) or (i == l):
            continue
          if (j==k) or (j==l):
            continue
          if (k==l):
            continue
          ## find the remaining numbers in the set
          comp1 = list(set(w) - set([i,j,k,l])) ## 5 remain

          ## get all the (abcd x e = fghi)
          for p in comp1:
            d2 = int(i+j+k+l)
            d1 = int(p)
            if d2 % d1 == 0:
              comp2 = list(set(comp1) - set([p])) ## 4 remain
              for s in comp2:
                for t in comp2:
                  for u in comp2:
                    for v in comp2:
                      if(s==t) or (s==u) or (s==v):
                        continue
                      if(t==u) or (t==v) or (u==v):
                        continue
                      d3 = int(s+t+u+v)
                      if int(d2/d1) == d3:
                        if d2 not in sprod:
                          sprod.append(d2)
                        perm.append([d2, d1, d3])
          ## get all the (abc x de = fghi)
          for p in comp1:
            for q in comp1:
              if (p == q):
                continue
              d1 = int(p+q)
              d2 = int(i+j+k+l)
              if (d2 % d1) == 0:
                comp2 = list(set(comp1) - set([p,q])) ## 3 remain
                for s in comp2:
                  for t in comp2:
                    for u in comp2:
                      if (s==t) or (s==u) or (t==u):
                        continue
                      d3 = int(s+t+u)
                      if int(d2/d1) == d3:
                        if d2 not in sprod:
                          sprod.append(d2)
                        perm.append([d2, d1, d3])
  #print (perm, sprod, sum(sprod))
  outp += "\nAnswer: " + str(sum(sprod)) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0035():
  outp = """36) How many circular primes are there below one million?"""
  cnt = 1 ## 2 is a prime
  result = [2] ## 2 is a prime

  primearray = [] ## alread exists array

  for i in range(3,1000000,2):
    narray = CircularNum(i)
    if narray == None:
      continue
    else:
      found = True
      for j in narray:
        if j in primearray:
          found = False
          break
        if not isPrime(j):
          found = False
          break
        else:
          primearray.append(j)
      if found:
        result = result + narray
        cnt = cnt + len(narray) 
   
  #print ("answr = " , result, len(result))
  outp += "\nAnswer: " + str(len(result)) + '\n'
  print (outp,"\n========================================================")
  

#__________________________________________________________________________________________________
def Problem0034():
  outp = """34) Find the sum of all numbers which are equal to the sum of the factorial of their digits."""
  sump = 0
  facarray = [1]
  for i in range(1,10):
    facarray.append(Factorial(i))
  for i in range(9,100000):
    arr = DigitsNum(i)[::-1]
    res = 0
    for j in range(0,len(arr)):
      res = res + facarray[arr[j]]
    if res == i:
      sump = res + sump
  outp += "\nAnswer: " + str(sump) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0036():
  outp = """36) Find the sum of all numbers, less than one million, which are palindromic in base 10 and base 2."""
  sumi = 0;
  for i in range(1,10):
    b2 = Base10ToBase2(i)
    if istrPalindrome(Base10ToBase2(i)):
      #print (i,b2)
      sumi = sumi + i

  for i in range(11,1000001,2):
    if inumPalindrome(i):
      b2 = Base10ToBase2(i)
      if istrPalindrome(b2):
        #print (i, b2)
        sumi = sumi + i
  outp += "\nAnswer: " + str(sumi) + '\n'
  print (outp,"\n========================================================")


#__________________________________________________________________________________________________
def Problem0039():
  outp = """If p is the perimeter of a right angle triangle with integral length sides, {a,b,c}, for which value of p <= 1000, is the number of solutions maximised?"""
  nsol_p = {}
  Nmin = 1
  Nmax = 1000
  outf = open("prob39.out.txt", 'w')
  for p in range(Nmin,Nmax):
    sols_p = []
    for b in range(Nmin,int(p/2)):
      for c in range(Nmin,int(p/2)):
        s = p**2 + 2*b*c - 2*p*(b+c)
        if s == 0:
          t = sorted([b,c,p-b-c])
          found = False
          for it in sols_p:
            if set(t).issubset(it):
              found = True
              break
          if not found:
            sols_p.append(t)
    nsol_p[p] = len(sols_p)
    print("%d: %d: " % (p, len(sols_p)), sols_p, end="\n", file=outf)
  outf.close()

  max_s = -1; max_i = -1
  for i in range(Nmin,Nmax):
    val = nsol_p[i]
    if val > max_s:
      max_s = val
      max_i = i
    #print("%d  %d" % (i,nsol_p[i]), end="\n", file=outf)
  outp += "\nAnswer: p = " + str(max_i) + ' has ' + str(max_s) + ' solutions'
  print (outp,"\n========================================================")


#__________________________________________________________________________________________________
def Problem0042(filen):
  lines = [(line.strip()).split(",") for line in open(filen)]
  val={}
  delta = 64  ## ASCII of 'A' = 65
  for i in range(1,27):
    val[chr(i+delta)] = i
  
  triset = []
  for i in range(1,20):
    t = int(i*(i+1)/2)
    triset.append(t)
  triwords = []
  wordvals = {}
  for word in lines[0]:
    w = word.strip("\"")
    tot = 0
    for i in range(len(w)):
      tot += val[w[i]]
    wordvals[w]=tot
    if tot in triset:
      triwords.append(w)

  print(triwords)
  outp = """The nth term of the sequence of triangle numbers is given by, t_n = n(n+1)/2; how many are triangle words? """
  outp += "\nAnswer: " + str(len(triwords))
  print (outp,"\n========================================================")

 

#__________________________________________________________________________________________________
def Problem0044():
  pentaset = []
  for i in range(1,15000):
    t = int(i*(3*i-1)/2)
    pentaset.append(t)

  for j in pentaset:
    for k in pentaset:
      if j < k:
        continue
      s = j + k
      d = j - k
      found_s = (s in pentaset)
      found_d = (d in pentaset)
      if found_s and found_d:
        print (j,k,s,d) #7042750 1560090 8602840 5482660
        break

#__________________________________________________________________________________________________
def Problem0045():
  outp = """T(285) = P(165) = H(143) = 40755. Find the next triangle number that is also pentagonal and hexagonal"""
  triset = []
  pentaset = []
  hexaset = []
  nmin=20000
  nmax=2*nmin
  for i in range(nmin,nmax-1):
    triset.append(int(i*(i+1)/2))
    pentaset.append(int(i*(3*i-1)/2))
    hexaset.append(int(i*(2*i-1)))

  phc = None
  for p in range(len(pentaset)):
    for h in range(len(hexaset)):
      if (pentaset[p] == hexaset[h]):
        phc = [p,h,pentaset[p]]
        break
  #print (phc) #[[11977, 7693, 1533776805]]
  if phc:
    phc.append(int((-1 + math.sqrt(1+4*phc[2]*2))/2))
  outp += "\nAnswer: " + str(phc[2])
  print (outp,"\n========================================================")



#__________________________________________________________________________________________________
def Problem0046():
  outp = """50) What is the smallest odd composite that cannot be written as the sum of a prime and twice a square?"""
  lp = ListOfPrimesBelow(5500)
  num = 5701
  while True:
    if not (num in lp):
      if not GoldBachConj(num,lp):
        outp += "\nAnswer: " + str(num) + '\n'
        break;
    num += 2
    if num > 6000:
      break
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0048():
  outp = """48) Find the last ten digits of the series 1^1 + 2^2 + 3^3 + ... + 1000^1000"""
  sumi = 0
  for j in range(1,1000):
    resn = j
    for i in range(1,j):
      resn = resn * j 
    sumi = sumi + resn
  sumi = str(sumi)
  l = len(sumi)
  outp += "\nAnswer: " + sumi[l-11:l-1] + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0049():
  outp = """What 12-digit number do you form by concatenating the three terms in this sequence?\nAnswer: """
  q = []
  final = []
  for i in range(1001,9999,2):
    if isPrime(i):
      w = str(i)
      s = [ w[0], w[1], w[2],w[3] ]
      if not (('0' in s) or ('4' in s) or ('8' in s) or ('7' in s) or ('1' in s)):
        q.append(s)

  ## get two numbers that are permutations of each other
  for i in range(len(q)):
    if len(final) == 1:
      break
    for j in range(len(q)):
      if i == j: 
        continue
      if len(final) == 1:
        break
      if sorted(q[i]) == sorted(q[j]):
        if len(final) == 1:
          break
        ## we found two, see if there is a third one in the list
        for k in range(len(q)):
          if k == i or k == j:
            continue
          ## we found 3, now check if they have same delta's
          if sorted(q[k]) == sorted(q[i]):
            n1 = int(''.join(q[i]))
            n2 = int(''.join(q[j]))
            n3 = int(''.join(q[k]))
            n = sorted([n1, n2, n3])
            if (n[1] - n[0]) == (n[2] - n[1]):
              if n not in final:
                final.append(n)
              break
  for x in final[0]:
    outp += ''.join(str(x))
  print (outp,"\n========================================================")


#__________________________________________________________________________________________________
def Problem0050():
  outp = """50) Which prime, below one-million, can be written as the sum of the most consecutive primes?"""
  lp = ListOfPrimesBelow(4000)
  j = 0
  for i in range(3, len(lp)):
    lpi = lp[3:i]
    sumpr = SumPrimes(lpi)
    if sumpr % 2 == 0:
      continue
    else:
      if isPrime(sumpr):
        if sumpr > 1000000:
          j = i-1
          break
  outp += "\nAnswer: " + str(SumPrimes(lp[3:j-1])) + '\n'
  print (outp,"\n========================================================")


#__________________________________________________________________________________________________
def Problem0063():
  outp = """63) How many n-digit positive integers exist which are also an nth power?"""
  nitems = 0
  for i in range(1,500):
    for j in range(1,500):
      a = i**j
      if len(str(a)) == j:
        nitems += 1
  outp += "\nAnswer: " + str(nitems) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0065():
  t = [2, 3]
  xf = [2, 1, 1]
  outp = """65) Find the sum of digits in the numerator of the 100th convergent of the continued fraction for e?"""
  for i in range(0,3*33,3):
    for j in range(0,3):
      k = 1
      if j == 0:
        k = int(i/3) + 1
      t.append ( t[i+1+j] * k * xf[j] + t[i+j] )
    if len(t) > 100:
      break
  n = t[100-1]
  s = 0
  for c in str(n):
    s += int(c)
  outp += "\nAnswer: " + str(s) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def Problem0092():
  t = []
  outp = """92) How many starting numbers below ten million will arrive at 89?"""
  for i in range(1,10000000):
    v = [i]
    cnt = 0
    while True:
      s = 0;
      for c in str(i):
        s = s + int(c)**2
      if (s == 89 or s == 1) and cnt > 0:
        if s == 89:
          v.append(cnt)
          t.append(v)
        break
      else:
        i = s
        cnt = cnt + 1
  outp += "\nAnswer: " + str(len(t)) + '\n'
  print (outp,"\n========================================================")

#__________________________________________________________________________________________________
def PowerDigitSum(filen):
  lines = [line.strip() for line in open(filen)]
  suml = 0
  #ans: 5537376230
  #0:08 = 553 73  76 181
  #1:09 =  47 73  76 22 56
  #2:10 =   493   76 22 992
  #3:11 =    5 0  76 230 3 42
  #4:12 =     55   6 230 3 860
  suml = [0,0,0,0,0]
  for j in range(0,5):
    for l in lines:
      suml[j] = suml[j] + int(l[j:j+8])
    print (suml[j])

#__________________________________________________________________________________________________
def probunknown():
  lycnt = 0
  lycarr = []
  #for num in range(10,10000):
  for num in range(0,2):
    num = 10677
    tnum = num
    itr = 0
    while itr < 57:
      res = tnum + ReverseNum(tnum)
      print (res)
      if inumPalindrome(res):
        lycarr.append([num,itr,res])
        lycnt = lycnt + 1
        break
      else:
        tnum = res
      itr = itr + 1
  print (lycnt,lycarr)


#__________________________________________________________________________________________________
def main(options, args, parser):
  Problem0092()
            

#__________________________________________________________________________________________________
if __name__ == "__main__":
    options, args,parser = proc_cmdline()
    sys.exit(main(options,args,parser))
