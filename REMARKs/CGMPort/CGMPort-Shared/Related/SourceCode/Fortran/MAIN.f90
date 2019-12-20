PROGRAM cgm99
IMPLICIT NONE
! NOTE: this code is written in a relatively clean structure, exploiting very few techniques for improving efficiency
! for this model it runs extremely fast so that is not an issue (but for more complicated setting this structure can become very slow)


!!!!!!!!!!!!!!!!!!!!!!
! DEFINING VARIABLES !
!!!!!!!!!!!!!!!!!!!!!!

CHARACTER(LEN=6), DIMENSION(80,1) :: filename
INTEGER :: tb, tr, td, tn, tt, t, nc_r, nalfa_r
parameter (tb=20, tr=65, td=100)
INTEGER :: ind1,ind2,ind4,ind5,lowc2,highc2,lowalfa2,highalfa2,aux2
INTEGER, DIMENSION(1) :: pt
INTEGER :: nqp, nalfa, ncash, nc
parameter (nqp=3, nalfa=101, ncash=401, nc=1501)
REAL :: delta=0.96, gamma= 10.0, infinity=-1e+10
REAL :: rf=1.02, sigma_r=0.157**2, exc= 0.04, mu, reg_coef=0.0 ! 0.0700855
REAL :: lowc, highc, lowalfa, highalfa, avg, ret_y=0.0, ret_fac=0.68212
REAL :: a=-2.170042+2.700381, b1=0.16818, b2=-0.0323371/10, b3=0.0019704/100
REAL :: sigt_y=0.0738,sigp_y=0.01065
REAL, DIMENSION(td-tb+1,1) :: survprob=0.0 , delta2=0.0
REAL, DIMENSION(nqp,1) :: grid,weig,gr,eyp,eyt,gret,ones_nqp_1=1.0
REAL, DIMENSION(nqp,1) :: expeyp
REAL, DIMENSION(nqp,tr-1) :: f_y
REAL, DIMENSION(1,ncash) :: gcash, ut
REAL, DIMENSION(ncash,1) :: aux3, secd
REAL, DIMENSION(1,nc) :: gc, u
REAL, DIMENSION(nalfa,1) :: galfa
REAL, DIMENSION(2,ncash) :: c, alfa=0.0, v
REAL, ALLOCATABLE :: gc_r(:,:), galfa_r(:,:), invest(:,:), u_r(:,:), u2(:,:), u3(:,:), nw(:,:)
REAL, ALLOCATABLE :: nv(:,:), v1(:,:), vv(:,:), auxV(:,:), ones(:,:)

filename(1,1) = 'year01'
filename(2,1) = 'year02'
filename(3,1) = 'year03'
filename(4,1) = 'year04'
filename(5,1) = 'year05'
filename(6,1) = 'year06'
filename(7,1) = 'year07'
filename(8,1) = 'year08'
filename(9,1) = 'year09'
filename(10,1) = 'year10'
filename(11,1) = 'year11'
filename(12,1) = 'year12'
filename(13,1) = 'year13'
filename(14,1) = 'year14'
filename(15,1) = 'year15'
filename(16,1) = 'year16'
filename(17,1) = 'year17'
filename(18,1) = 'year18'
filename(19,1) = 'year19'
filename(20,1) = 'year20'
filename(21,1) = 'year21'
filename(22,1) = 'year22'
filename(23,1) = 'year23'
filename(24,1) = 'year24'
filename(25,1) = 'year25'
filename(26,1) = 'year26'
filename(27,1) = 'year27'
filename(28,1) = 'year28'
filename(29,1) = 'year29'
filename(30,1) = 'year30'
filename(31,1) = 'year31'
filename(32,1) = 'year32'
filename(33,1) = 'year33'
filename(34,1) = 'year34'
filename(35,1) = 'year35'
filename(36,1) = 'year36'
filename(37,1) = 'year37'
filename(38,1) = 'year38'
filename(39,1) = 'year39'
filename(40,1) = 'year40'
filename(41,1) = 'year41'
filename(42,1) = 'year42'
filename(43,1) = 'year43'
filename(44,1) = 'year44'
filename(45,1) = 'year45'
filename(46,1) = 'year46'
filename(47,1) = 'year47'
filename(48,1) = 'year48'
filename(49,1) = 'year49'
filename(50,1) = 'year50'
filename(51,1) = 'year51'
filename(52,1) = 'year52'
filename(53,1) = 'year53'
filename(54,1) = 'year54'
filename(55,1) = 'year55'
filename(56,1) = 'year56'
filename(57,1) = 'year57'
filename(58,1) = 'year58'
filename(59,1) = 'year59'
filename(60,1) = 'year60'
filename(61,1) = 'year61'
filename(62,1) = 'year62'
filename(63,1) = 'year63'
filename(64,1) = 'year64'
filename(65,1) = 'year65'
filename(66,1) = 'year66'
filename(67,1) = 'year67'
filename(68,1) = 'year68'
filename(69,1) = 'year69'
filename(70,1) = 'year70'
filename(71,1) = 'year71'
filename(72,1) = 'year72'
filename(73,1) = 'year73'
filename(74,1) = 'year74'
filename(75,1) = 'year75'
filename(76,1) = 'year76'
filename(77,1) = 'year77'
filename(78,1) = 'year78'
filename(79,1) = 'year79'
filename(80,1) = 'year80'

!!!!!!!!!!!!!!
! QUADRATURE !
!!!!!!!!!!!!!!

 weig(1,1)= 0.1666666666666
 weig(2,1)= 0.6666666666666
 weig(3,1)= 0.1666666666666
 grid(1,1)= -1.73205080756887
 grid(2,1)=  0.0
 grid(3,1)=  1.73205080756887

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! CONDITIONAL SURVIVAL PROBABILITIES  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

survprob(1,1) = 0.99845
survprob(2,1) = 0.99839
survprob(3,1) = 0.99833
survprob(4,1) = 0.9983
survprob(5,1) = 0.99827
survprob(6,1) = 0.99826
survprob(7,1) = 0.99824
survprob(8,1) = 0.9982
survprob(9,1) = 0.99813
survprob(10,1) = 0.99804
survprob(11,1) = 0.99795
survprob(12,1) = 0.99785
survprob(13,1) = 0.99776
survprob(14,1) = 0.99766
survprob(15,1) = 0.99755
survprob(16,1) = 0.99743
survprob(17,1) = 0.9973
survprob(18,1) = 0.99718
survprob(19,1) = 0.99707
survprob(20,1) = 0.99696
survprob(21,1) = 0.99685
survprob(22,1) = 0.99672
survprob(23,1) = 0.99656
survprob(24,1) = 0.99635
survprob(25,1) = 0.9961
survprob(26,1) = 0.99579
survprob(27,1) = 0.99543
survprob(28,1) = 0.99504
survprob(29,1) = 0.99463
survprob(30,1) = 0.9942
survprob(31,1) = 0.9937
survprob(32,1) = 0.99311
survprob(33,1) = 0.99245
survprob(34,1) = 0.99172
survprob(35,1) = 0.99091
survprob(36,1) = 0.99005
survprob(37,1) = 0.98911
survprob(38,1) = 0.98803
survprob(39,1) = 0.9868
survprob(40,1) = 0.98545
survprob(41,1) = 0.98409
survprob(42,1) = 0.9827
survprob(43,1) = 0.98123
survprob(44,1) = 0.97961
survprob(45,1) = 0.97786
survprob(46,1) = 0.97603
survprob(47,1) = 0.97414
survprob(48,1) = 0.97207
survprob(49,1) = 0.9697
survprob(50,1) = 0.96699
survprob(51,1) = 0.96393
survprob(52,1) = 0.96055
survprob(53,1) = 0.9569
survprob(54,1) = 0.9531
survprob(55,1) = 0.94921
survprob(56,1) = 0.94508
survprob(57,1) = 0.94057
survprob(58,1) = 0.9357
survprob(59,1) = 0.93031
survprob(60,1) = 0.92424
survprob(61,1) = 0.91717
survprob(62,1) = 0.90922
survprob(63,1) = 0.90089
survprob(64,1) = 0.89282
survprob(65,1) = 0.88503
survprob(66,1) = 0.87622
survprob(67,1) = 0.86576
survprob(68,1) = 0.8544
survprob(69,1) = 0.8423
survprob(70,1) = 0.82942
survprob(71,1) = 0.8154
survprob(72,1) = 0.80002
survprob(73,1) = 0.78404
survprob(74,1) = 0.76842
survprob(75,1) = 0.75382
survprob(76,1) = 0.73996
survprob(77,1) = 0.72464
survprob(78,1) = 0.71057
survprob(79,1) = 0.6961
survprob(80,1) = 0.6809

delta2 = delta*survprob

!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ADDITIONAL COMPUTATIONS !
!!!!!!!!!!!!!!!!!!!!!!!!!!!

tn = td-tb+1

gr=grid*sigma_r**0.5
eyp=grid*sigp_y**0.5
eyt=grid*sigt_y**0.5
mu = exc+rf

expeyp = EXP(eyp)

!!!!!!!!!!!!!!!!!!!
! CONSTRUCT GRIDS !
!!!!!!!!!!!!!!!!!!!

DO ind1=1,nalfa
   galfa(ind1,1)=(ind1-1.0)/(nalfa-1.0)
END DO
gret = mu*ones_nqp_1+gr
DO ind1=1,ncash
   gcash(1,ind1)=4.0+(ind1-1.0)*1.0
END DO
aux3(:,1) = gcash(1,:)
DO ind1=1,nc
   !gc(1,ind1)=0.0+(ind1-1.0)*0.25
   gc(1,ind1)=1+(ind1-1.0)*0.25
END DO

!!!!!!!!!!!!!!!!
! LABOR INCOME !
!!!!!!!!!!!!!!!!


DO ind1=tb+1,tr
   avg = EXP(a+b1*ind1+b2*ind1**2+b3*ind1**3)
   f_y(:,ind1-tb) = avg*EXP(eyt(:,1))
END DO


ret_y= ret_fac*avg



!!!!!!!!!!!!!!!!!!!
! TERMINAL PERIOD !
!!!!!!!!!!!!!!!!!!!
CALL utility(gcash,ncash,gamma,ut)

v(1,:)= ut(1,:)
c(1,:)= gcash(1,:)



!!!!!!!!!!!!!!!!!!!!!!
! RETIREMENT PERIODS !
!!!!!!!!!!!!!!!!!!!!!!
CALL utility(gc,nc,gamma,u)
tt=80
DO ind1= 1,35
   t= tt-ind1+1
   WRITE(*,*) t
   call spline(aux3,v(1,:),ncash,gamma,secd)
   DO ind2= 1,ncash
      IF (t.eq.tn-1) THEN
         lowc= c(1,ind2)/2.0
         highc= c(1,ind2)
         if (gcash(1,ind2).ge.50) then
             highc= c(1,ind2)/1.5
         end if
       ELSE IF (t.eq.tn-2) THEN
         lowc= c(1,ind2)/2.5
         highc= c(1,ind2)
         if (gcash(1,ind2).ge.50) then
             highc= c(1,ind2)/1.2
         end if
       ELSE IF (t<tn-2 .AND. t>tn-5) THEN
         lowc= c(1,ind2)/3.5
         highc= c(1,ind2)+0.0
         if (gcash(1,ind2).ge.50) then
            highc= c(1,ind2)/1.1
         end if
       ELSE
         lowc= c(1,ind2)-10.0
         highc= c(1,ind2)+10.0
       END IF
       CALL ntoi(lowc,1,gc,nc,lowc2)
       CALL ntoi(highc,1,gc,nc,highc2)
       nc_r= highc2-lowc2+1
       ALLOCATE (gc_r(1,nc_r))
       gc_r(1,:)= gc(1,lowc2:highc2)
       lowalfa2= 1.0
       highalfa2= nalfa
       IF (gcash(1,ind2)>40.0.and.t<tn-1) THEN
          lowalfa= alfa(1,ind2)-0.2
          highalfa= alfa(1,ind2)+0.2
          CALL ntoi(lowalfa,1,galfa,nalfa,lowalfa2)
          CALL ntoi(highalfa,1,galfa,nalfa,highalfa2)
       END IF
       nalfa_r= highalfa2-lowalfa2+1
       ALLOCATE(galfa_r(nalfa_r,1))
       galfa_r(:,1) = galfa(lowalfa2:highalfa2,1)
       ALLOCATE(invest(nc_r,1))
       ALLOCATE(ones(nc_r,1))
       ones(:,1) = 1.0
       invest(:,1) = gcash(1,ind2)*ones(:,1)-gc_r(1,:)
       DEALLOCATE(ones)
       ALLOCATE(u_r(nc_r,1))
       u_r(:,1) = u(1,lowc2:highc2)
       ALLOCATE(u2(nc_r,1))
       do ind4=1,nc_r
          IF (invest(ind4,1)<0.0) then
            u2(ind4,1) = infinity
      else
        u2(ind4,1) = u_r(ind4,1)
      END if
       end do
       invest = MAX(invest,0.0)
       ALLOCATE(u3(nc_r,nalfa_r))
       do ind4=1,nalfa_r
          u3(:,ind4)=u2(:,1)
       end do
       u3 = MAX(u3,infinity)
       ALLOCATE(v1(nc_r,nalfa_r))
       v1=0.0
       ALLOCATE(nw(nc_r,nalfa_r))
       ALLOCATE(nv(nc_r,nalfa_r))
       do ind5=1,nqp
          call fci(invest,nc_r,galfa_r,nalfa_r,gret(ind5,1),rf,nw)
          call evr(nw,nc_r,nalfa_r,v(1,:),1,ncash,ret_y,aux3,secd,nv)
          v1 = v1+nv*weig(ind5,1)
       end do
       DEALLOCATE(nw,nv,u2,u_r,invest,galfa_r,gc_r)
       ALLOCATE(vv(nc_r,nalfa_r))
       vv = u3+delta2(t,1)*v1
       vv = MAX(vv,infinity)
       ALLOCATE(auxv(nc_r*nalfa_r,1))
       auxv = RESHAPE((vv),(/nc_r*nalfa_r,1/))
       v(2,ind2) = MAXVAL(auxv(:,1))
       pt = MAXLOC(auxv(:,1))
       aux2 = FLOOR((REAL(pt(1))-1)/REAL(nc_r))
       alfa(2,ind2) = galfa(aux2+lowalfa2,1)
       c(2,ind2) = gc(1,pt(1)-aux2*nc_r+lowc2-1)
       DEALLOCATE(auxv,vv,v1,u3)
  END DO
  OPEN(UNIT=25,FILE=filename(t,1),STATUS='replace',ACTION='write')
  do ind5=1,ncash
    WRITE(25,*) alfa(2,ind5)
  end do
  do ind5=1,ncash
    WRITE(25,*) c(2,ind5)
  end do
  do ind5=1,ncash
    WRITE(25,*) v(2,ind5)
  end do
  CLOSE(UNIT=25)
  v(1,:)=v(2,:)
  c(1,:)=c(2,:)
  alfa(1,:)=alfa(2,:)
END DO

!!!!!!!!!!!!!!!!!
! OTHER PERIODS !
!!!!!!!!!!!!!!!!!


DO ind1= 1,tt-35
   t= 45-ind1+1
   WRITE(*,*) t
   call spline(aux3,v(1,:),ncash,gamma,secd(:,1))
     DO ind2= 1,ncash
	IF (t<tr-19 .AND. t>tr-25) THEN
	   lowc= c(1,ind2)-10.0
	   highc= c(1,ind2)+10.0
	ELSE
	   lowc= c(1,ind2)-5.0
	   highc= c(1,ind2)+5.0
	END IF
	CALL ntoi(lowc,1,gc,nc,lowc2)
	CALL ntoi(highc,1,gc,nc,highc2)
	nc_r= highc2-lowc2+1
	ALLOCATE (gc_r(1,nc_r))
	gc_r(1,:)= gc(1,lowc2:highc2)
	lowalfa2= 1.0
	highalfa2= nalfa
	IF (gcash(1,ind2)>40.0.and.t<tn-1) THEN
	   lowalfa= alfa(1,ind2)-0.2
	   highalfa= alfa(1,ind2)+0.2
	   CALL ntoi(lowalfa,1,galfa,nalfa,lowalfa2)
	   CALL ntoi(highalfa,1,galfa,nalfa,highalfa2)
	END IF
	nalfa_r= highalfa2-lowalfa2+1
	ALLOCATE(galfa_r(nalfa_r,1))
	galfa_r(:,1) = galfa(lowalfa2:highalfa2,1)
	ALLOCATE(invest(nc_r,1))
	ALLOCATE(ones(nc_r,1))
	ones(:,1) = 1.0
	invest(:,1) = gcash(1,ind2)*ones(:,1)-gc_r(1,:)
	DEALLOCATE(ones)
	ALLOCATE(u_r(nc_r,1))
	u_r(:,1) = u(1,lowc2:highc2)
	ALLOCATE(u2(nc_r,1))
	do ind4=1,nc_r
	   IF (invest(ind4,1)<0.0) then
	      u2(ind4,1) = infinity
	   else
	      u2(ind4,1) = u_r(ind4,1)
	   END if
	end do
	invest = MAX(invest,0.0)
	ALLOCATE(u3(nc_r,nalfa_r))
	do ind4=1,nalfa_r
	   u3(:,ind4)=u2(:,1)
	end do
	u3 = MAX(u3,infinity)
	ALLOCATE(v1(nc_r,nalfa_r))
	v1=0.0
	ALLOCATE(nw(nc_r,nalfa_r))
	ALLOCATE(nv(nc_r,nalfa_r))
	do ind5=1,nqp
	   call fci(invest,nc_r,galfa_r,nalfa_r,gret(ind5,1),rf,nw)
	   call ev(nw,nc_r,nalfa_r,v(1,:),ncash,weig,nqp,f_y(:,t),expeyp,aux3,secd,gret(ind5,1),reg_coef,nv) !Former multiline comment. Changed.
	   v1 = v1+nv*weig(ind5,1)
	end do
	DEALLOCATE(nw,nv,u2,u_r,invest,galfa_r,gc_r)
	ALLOCATE(vv(nc_r,nalfa_r))
	vv = u3+delta2(t,1)*v1
	vv = MAX(vv,infinity)
	ALLOCATE(auxv(nc_r*nalfa_r,1))
	auxv = RESHAPE((vv),(/nc_r*nalfa_r,1/))
	v(2,ind2) = MAXVAL(auxv(:,1))
	pt = MAXLOC(auxv(:,1))
	aux2 = FLOOR((REAL(pt(1))-1)/REAL(nc_r))
	alfa(2,ind2) = galfa(aux2+lowalfa2,1)
	c(2,ind2) = gc(1,pt(1)-aux2*nc_r+lowc2-1)
	DEALLOCATE(auxv,vv,v1,u3)
   END DO
   OPEN(UNIT=25,FILE=filename(t,1),STATUS='replace',ACTION='write')
      do ind5=1,ncash
	 WRITE(25,*) alfa(2,ind5)
      end do
      do ind5=1,ncash
	 WRITE(25,*) c(2,ind5)
      end do
      do ind5=1,ncash
	 WRITE(25,*) v(2,ind5)
      end do
   CLOSE(UNIT=25)
   v(1,:)=v(2,:)
   c(1,:)=c(2,:)
   alfa(1,:)=alfa(2,:)
END DO





end program

