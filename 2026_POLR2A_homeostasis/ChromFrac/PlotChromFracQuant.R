library(tidyverse)

chrom_frac <- read.csv("/Users/z3532965/src/publications/2026_POLR2A_homeostasis/ChromFrac/20260616_Berrylab_chromFrac_quant_norm_to_loading.csv")

REPLICATE_COL <- "Replicate"

chrom_frac <- chrom_frac %>%
  pivot_longer(-c("DRUG_TIME_HOURS","Target")) %>%
  separate(name,into=c("Fraction","Replicate"),sep="_") %>%
  mutate(Target = if_else(Target=="3E10","pSer2",Target),
         Target = if_else(Target=="D8L4Y","Total",Target))

chrom_frac_normalisers <- chrom_frac %>%
  filter(DRUG_TIME_HOURS==0) %>%
  select(-DRUG_TIME_HOURS) %>%
  rename(value_t0 = value)

chrom_frac %>%
  left_join(chrom_frac_normalisers) %>%
  mutate(value_norm = value / value_t0) %>%
  group_by(DRUG_TIME_HOURS,Target,Fraction) %>%
  filter(!(Target=="pSer2" & Fraction=="Soluble")) %>%
  mutate(Target=factor(Target,levels=c("mClover","mCherry","Total","pSer2","CDK9","SPT5")),
         Fraction=factor(Fraction,levels=c("Soluble","Chromatin"))) %>%
  ggplot(aes(x=Replicate,y=value_norm,fill=factor(DRUG_TIME_HOURS))) +
  geom_col(position = position_dodge()) +
  facet_grid(Fraction~Target) +
  scale_fill_manual(name="5-Ph-IAA duration (h)", values=c("grey","red","blue")) +
  scale_x_discrete() +
  scale_y_continuous(name="Normalised intensity\n(western blot quantification)", expand = expansion(mult = c(0, 0.05))) +
  theme_bw(8) +
  theme(#axis.text.x=element_blank(),
        #axis.title.x=element_blank(),
        panel.grid=element_blank(),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.key.size = unit(2,"mm"))
ggsave("/Users/z3532965/src/publications/2026_POLR2A_homeostasis/ChromFrac/fractionation_blot_quant_replicates.pdf",width=10,height=7,units="cm")


chrom_summary <- chrom_frac %>%
  left_join(chrom_frac_normalisers) %>%
  mutate(value_norm = value / value_t0) %>%
  group_by(DRUG_TIME_HOURS, Target, Fraction) %>%
  filter(!(Target == "pSer2" & Fraction == "Soluble")) %>%
  summarise(log_mean = mean(log(value_norm), na.rm = TRUE),
            log_sd   = sd(log(value_norm), na.rm = TRUE),
            .groups = "drop") %>%
  mutate(log_mean_lower = log_mean - log_sd,
         log_mean_upper = log_mean + log_sd) %>%
  mutate(mean       = exp(log_mean),
         mean_lower = exp(log_mean_lower),
         mean_upper = exp(log_mean_upper)) %>%
  mutate(Target   = factor(Target, levels = c("mClover","mCherry","Total","pSer2","CDK9","SPT5")),
         Fraction = factor(Fraction, levels = c("Soluble","Chromatin")))

points_df <- chrom_frac %>%
  left_join(chrom_frac_normalisers) %>%
  mutate(value_norm = value / value_t0) %>%
  filter(!(Target == "pSer2" & Fraction == "Soluble")) %>%
  mutate(Target   = factor(Target,   levels = levels(chrom_summary$Target)),
         Fraction = factor(Fraction, levels = levels(chrom_summary$Fraction)))

ggplot(chrom_summary, aes(x = 1, y = mean, fill = factor(DRUG_TIME_HOURS))) +
  geom_col(position = position_dodge(), alpha = 0.6) +
  geom_errorbar(aes(ymin = mean_lower, ymax = mean_upper, alpha=factor(DRUG_TIME_HOURS)),
                position = position_dodge(0.9), width = 0.5) +
  geom_jitter(data = points_df,
              aes(x = 1, y = value_norm, fill = factor(DRUG_TIME_HOURS),
                  group = factor(DRUG_TIME_HOURS),
                  shape = factor(DRUG_TIME_HOURS)),
              position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.9),
              colour = "black", size = 1.0, stroke = 0.3,
              alpha = 0.9, inherit.aes = FALSE) +
  scale_shape_manual(name = "5-Ph-IAA duration (h)",values = c(NA,21,21)) +
  scale_alpha_manual(name = "5-Ph-IAA duration (h)",values = c(0,0.5,0.5)) +
  facet_grid(Fraction ~ Target) +
  scale_fill_manual(name = "5-Ph-IAA duration (h)", values = c("grey","red","blue")) +
  scale_x_discrete() +
  scale_y_continuous(name = "Normalised intensity\n(western blot quantification)",
                     expand = expansion(mult = c(0, 0.05)),
                     breaks=c(0,1,2)) +
  theme_minimal(8) +
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_rect(linewidth = 0.75),
        panel.spacing = unit(1,"mm"),
        axis.ticks = element_line(linewidth = 0.5),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.key.size = unit(2, "mm"),
        strip.text = element_text(size=8))

ggsave("/Users/z3532965/src/publications/2026_POLR2A_homeostasis/ChromFrac/fractionation_blot_quant.pdf",width=9,height=5,units="cm")
 
# pSer2 has no Soluble measurement, so no ratio can be computed for it
ratio_raw <- chrom_frac %>%
  filter(Target != "pSer2" & Target != "mClover") %>%
  select(Target, Fraction, DRUG_TIME_HOURS, all_of(REPLICATE_COL), value) %>%
  pivot_wider(names_from = Fraction, values_from = value) %>%
  mutate(ratio = Chromatin / Soluble) %>%
  mutate(Target = factor(Target, levels = c("mClover","mCherry","Total","CDK9","SPT5")))

ratio_t0 <- ratio_raw %>%
  filter(DRUG_TIME_HOURS == 0) %>%
  select(Target, all_of(REPLICATE_COL), ratio_t0 = ratio)

ratio_df <- ratio_raw %>%
  left_join(ratio_t0, by = c("Target", REPLICATE_COL)) %>%
  mutate(ratio_norm = ratio / ratio_t0)

chrom_summary_ratio <- ratio_df %>%
  group_by(DRUG_TIME_HOURS, Target) %>%
  summarise(log_mean = mean(log(ratio_norm), na.rm = TRUE),
            log_sd   = sd(log(ratio_norm), na.rm = TRUE),
            .groups = "drop") %>%
  mutate(log_mean_lower = log_mean - log_sd,
         log_mean_upper = log_mean + log_sd) %>%
  mutate(mean       = exp(log_mean),
         mean_lower = exp(log_mean_lower),
         mean_upper = exp(log_mean_upper)) %>%
  mutate(Target = factor(Target, levels = c("mClover","mCherry","Total","CDK9","SPT5")))

ggplot(chrom_summary_ratio, aes(x = 1, y = mean, fill = factor(DRUG_TIME_HOURS))) +
  geom_col(position = position_dodge(), alpha = 0.6) +
  geom_errorbar(aes(ymin = mean_lower, ymax = mean_upper, alpha = factor(DRUG_TIME_HOURS)),
                position = position_dodge(0.9), width = 0.5) +
  geom_jitter(data = ratio_df,
              aes(x = 1, y = ratio_norm, fill = factor(DRUG_TIME_HOURS),
                  group = factor(DRUG_TIME_HOURS),
                  shape = factor(DRUG_TIME_HOURS)),
              position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.9),
              colour = "black", size = 1.3, stroke = 0.3,
              alpha = 0.9, inherit.aes = FALSE) +
  scale_shape_manual(name = "5-Ph-IAA duration (h)",values = c(NA,21,21)) +
  scale_alpha_manual(name = "5-Ph-IAA duration (h)",values = c(0,0.5,0.5)) +
  facet_wrap(~Target, nrow = 1) +
  scale_fill_manual(name = "5-Ph-IAA duration (h)", values = c("grey","red","blue")) +
  scale_x_discrete() +
  scale_y_continuous(name = "Chromatin / Soluble",
                     expand = expansion(mult = c(0, 0.05))) +
  theme_minimal(8) +
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_rect(linewidth = 0.75),
        panel.spacing = unit(1,"mm"),
        axis.ticks = element_line(linewidth = 0.5),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.key.size = unit(2, "mm"),
        strip.text = element_text(size=8))
ggsave("/Users/z3532965/src/publications/2026_POLR2A_homeostasis/ChromFrac/fractionation_blot_quant_ratio.pdf",width=6,height=3.5,units="cm")
