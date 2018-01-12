from collections import defaultdict
from gtnlplib import constants

theta_hand_original = defaultdict(float,
                         {('worldnews','worldnews'):1.,
                          ('worldnews','news'):.5,
                          ('worldnews','world'):.5,
                          ('science','science'):1.,
                          ('askreddit','askreddit'):1.,
                          ('askreddit','ask'):0.5,
                          ('iama','iama'):1,
                          ('todayilearned','til'):1.,
                          ('todayilearned','todayilearned'):1.,
                          ('iama',constants.OFFSET):0.1
                         })

# add some more weights to this for deliverable 2.3
theta_hand = defaultdict(float,
                         {('worldnews','worldnews'):1.,
                          ('worldnews','news'):.5,
                          ('worldnews','world'):.5,
                          ('science','science'):1.,
                          ('askreddit','askreddit'):1.,
                          ('askreddit','ask'):0.5,
                          ('iama','iama'):1,
                          ('todayilearned','til'):1.,
                          ('todayilearned','todayilearned'):1.,
                          ('iama',constants.OFFSET):0.1,
                          ('science','research'):.3,
                          ('science','method'):.3,
                          ('worldnews','china'):.3,
                          ('worldnews','russia'):.3,
                          ('worldnews','states'):.3,
                          ('worldnews','international'):.3,
                          ('todayilearned','funny'):.3
                         })
